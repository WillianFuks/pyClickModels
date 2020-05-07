import glob
import gzip
import time
import os
import ujson
from functools import partial
from multiprocessing import Manager, cpu_count, current_process
import random
from cpython cimport array
import array
import numpy as np


cdef class Factor:
    """
    Helper class to implement the Factor component as discussed in:

    https://clickmodels.weebly.com/uploads/5/2/2/5/52257029/mc2015-clickmodels.pdf

    page 37 equation 4.43

    Args
    ----
      doc: str
          Current analyzed document in session.
      click: int
      purchase: int
      dbn_params: multiprocessing.ProxyDict
          Last updated valeus of alpha, sigma and gamma.
      cr: float
          Conversion Rate of current document in session.
    """
    def __cinit__(
        self,
        int r,
        int last_r,
        str doc,
        str query,
        bint click,
        bint purchase,
        object dbn_params,
        float cr,
        array.array e_r_array_given_CP,
        array.array cp_array_given_e
    ):
        self.r = r
        self.last_r = last_r
        self.alpha = dbn_params.get(query, {}).get(doc, {}).get(
            'alpha', random.random())
        self.sigma = dbn_params.get(query, {}).get(doc, {}).get(
            'sigma', random.random())
        self.gamma = dbn_params.get('gamma', random.random())
        self.click = <bint>click
        self.purchase = <bint>purchase
        self.cr = cr
        self.e_r_array_given_CP = e_r_array_given_CP
        self.cp_array_given_e = cp_array_given_e

    cdef float compute_factor(self, bint x, bint y, bint z):
        """
        Responsible for computing the following equation:

        P(E_r = x, S_r = y, E_{r+1} = z, C_{>=r+1}, P_{>=r+1} | C_{<r}, P_{<r})
        """
        cdef:
            float result = 1
        # Compute P(E_{r+1}=z, S_r=y, C_r=c_r, P_r=p_r | E_r=x)
        if not x:
            if z or y or self.click:
                return 0
        else:
            if self.purchase:
                if not y:
                    return 0
                else:
                    if z:
                        return 0
                    else:
                        result *= self.alpha * self.cr * (1 - self.gamma) * self.sigma
            else:
                if not self.click:
                    if y:
                        return 0
                    else:
                        result *= (1 - self.alpha) * (1 - self.sigma)
                        if z:
                            result *= self.gamma
                        else:
                            result *= (1 - self.gamma)
                else:
                    result *= self.alpha
                    if not y:
                        result *= (1 - self.sigma)
                        if not z:
                            result *= (1 - self.gamma) * (1 - self.cr)
                        else:
                            result *= self.gamma * (1 - self.cr)
                    else:
                        result *= self.sigma
                        if not z:
                            result *= (1 - self.gamma) * (1 - self.cr)
                        else:
                            result *= self.gamma * (1 - self.cr)
        if not z:
            if self.last_r >= self.r + 1:
                return 0
        else:
            if self.r < len(self.cp_array_given_e):
                result *= self.cp_array_given_e[self.r]
        result *= (self.e_r_array_given_CP[self.r] if x else
                   1 - self.e_r_array_given_CP[self.r])
        return result


cdef class DBNModel():

    def run_EM(self, str file_name, object cr_dict):
        """
        Reads through all input files and iterates through values running the EM
        algorithm as defined by the DBN model. On each iteration, values in `dbn_params`
        are updated.

        Args
        ----
          file_name: str
          dbn_params: multiprocessing.DictProxy
          cr_dict: multiprocessing.DictProxy
        """
        cdef:
            bytes row
            dict json_row
            list sessions
            dict session
            dict tmp_vars

        print(file_name)
        c = 0

        for row in gzip.GzipFile(file_name):
            print(c)
            c += 1
            # As Python dictionaries are already quite performant we use them whenever
            # possible
            tmp_vars = {}
            tmp_vars['gamma'] = [1, 2]
            t0 = time.time()
            json_row = ujson.loads(row)
            print('json row time: ', time.time() - t0)
            t0 = time.time()
            query = self.get_search_context_string(json_row['search_keys'])
            print('query time: ', time.time() - t0)
            sessions = json_row['judgment_keys']
            self.compute_cr(query, sessions, cr_dict)
            for session in sessions:
                t0 = time.time()
                self.update_params(session, tmp_vars, query, cr_dict[query])
                print('update_params time: ', time.time() - t0)
            t0 = time.time()
            self.update_dbn_params(query, tmp_vars)
            print('update dbn params time: ', time.time() - t0)


    cdef void update_dbn_params(self, str query, dict tmp_vars):
        """
        After all sessions for a given query have been analyzed, the new values of
        alpha, sigma and gamma in `tmp_vars` are copied into `dbn_params` where they'll
        be used into new optimization iterations.

        Args
        ----
          query: str
          dbn_params: multiprocessing.ProxyDict
              Dictionary with current values of alpha, sigma and gamma.
          tmp_vars: dict
              Dictionary with updated values of alpha, sigma and gamma after an EM
              iteration round.
        """
        cdef dict tmp = {}

        self.dbn_params['gamma'] = tmp_vars['gamma'][0] / tmp_vars['gamma'][1]
        # After updating persistence we no longer need it
        tmp_vars.pop('gamma')

        for doc in tmp_vars:
            for var in tmp_vars[doc]:
                tmp[var] = tmp_vars[doc][var][0] / tmp_vars[doc][var][1]
            self.dbn_params[query] = {doc: tmp}
        print('dbn: ', self.dbn_params.items())

    cdef void update_params(
        self,
        dict session,
        dict tmp_vars,
        str query,
        object cr_dict
    ):
        """
        For each session, applies the EM algorithm and save temporary results into
        `tmp_vars`.

        Args
        ----
          sessions: dict
              Dict containing documents users observed on search results page and their
              interaction with each item. Example:
                `{"session ID 1": [{"doc": "doc0", "click": 0, "purchase": 0}]}`
          tmp_vars: dict
              Holds temporary values for adapting each variable \\alpha, \\sigma and
              \\gamma.
          query: str
          dbn_params: multiprocessing.ProxyDict
              Holds updated values of \\alpha, \\sigma and \\gamma. When all iterations
              are completed, those values are the converged expectation for each
              parameter.
          cr_dict: multiprocessing.DictProxy
              Conversion Rates of each document for the current query.
        """
        cdef:
            list s
            array.array e_r_array, X_r_array, e_r_array_given_CP, cp_array_given_e
            int last_r, r

        s = list(session.values())[0]
        t0 = time.time()
        e_r_array = self.build_e_r_array(s, query, cr_dict)
        print('e_r_array time: ', time.time() - t0)
        t0 = time.time()
        X_r_array = self.build_X_r_array(s, query)
        print('X_r_array time: ', time.time() - t0)
        t0 = time.time()
        e_r_array_given_CP = self.build_e_r_array_given_CP(s, query)
        print('e_r_array_given_CP time: ', time.time() - t0)
        t0 = time.time()
        cp_array_given_e = self.build_CP_array_given_e(s, query, cr_dict)
        print('cp_array_given_e time: ', time.time() - t0)
        last_r = self.get_last_r(s)

        for r in range(len(s)):
            t0 = time.time()
            self.update_alpha(r, query, s[r], e_r_array, X_r_array, last_r, tmp_vars)
            print('update_alpha time: ', time.time() - t0)
            t0 = time.time()
            self.update_sigma(query, r, s[r], X_r_array, last_r, tmp_vars)
            print('update_sigma time: ', time.time() - t0)
            t0 = time.time()
            self.update_gamma(r, last_r, s[r], query, cp_array_given_e,
                              e_r_array_given_CP, cr_dict, tmp_vars)
            print('update_gamma time: ', time.time() - t0)

    cdef array.array build_e_r_array(self, list session, str query,
                                     object cr_dict):
        """
        Computes the probability of each sku in user session being examined.

        The equation implemented is:

        $P(E_{r+1}=1) = \\epsilon_r \\gamma \\left((1 - \\alpha_{uq}) +
            (1 - \\sigma_{uq})(1 - cr_{uq})\\alpha_{uq} \\right)$

        Args
        ----
          session: List[Dict[str, Any]]
              List with all documents and interactions users had on search result pages.
          query: str
          dbn_params: multiprocessing.ProxyDict
              Contains updated values of \\alpha, \\sigma and \\gamma for each query
              and each document. Example:
              `{"query": {"doc0": {"alpha": 0.321, "sigma": 0.101}}, "gamma": 0.7}`
          cr_dict: Dict[str, double]
              Conversion Rates of each document for given query.

        Returns
        -------
          e_r_array: array.array
              Probability that each document was examined in current user session.
        """
        cdef:
            int r
            str doc
            double alpha
            double sigma
            double gamma
            double cr
            array.array e_r_next
            # First document in DBN is always considered to have been examined
            array.array e_r_array = array.array('f', [1])

        for r in range(len(session)):
            doc = session[r]['doc']
            alpha = self.dbn_params.get(query, {}).get(doc, {}).get('alpha', random.random())
            sigma = self.dbn_params.get(query, {}).get(doc, {}).get('sigma', random.random())
            gamma = self.dbn_params.get('gamma', random.random())
            cr = cr_dict[doc]
            e_r_next = array.array(
                'f',
                [e_r_array[r] * gamma * ((1 - sigma) * (1 - cr) * alpha + (1 - alpha))]
            )
            array.extend(e_r_array, e_r_next)
        return e_r_array

    cdef int get_last_r(self, list session, str event='click'):
        """
        Loops through all documents in session and find at which position the desired
        event happend. It can be either a 'click' or a 'purchase' (still, in DBN, if
        a purchase is observed then it automatically means it is the very last r
        observed.

        Args
        ----
          session: List[Dict[str, Any]]
              List with all documents and interactions users had on search result pages.
          event: str
              Name of desired event to track.

        Returns
        -------
          last_r: int
              Index at which the last desired event was observed.
        """
        cdef:
            int r
            int idx = 0
        for r in range(len(session)):
            if session[r][event] == 1:
                idx = r
        return idx

    cdef void update_gamma(
        self,
        int r,
        int last_r,
        dict doc_data,
        str query,
        array.array cp_array_given_e,
        array.array e_r_array_given_CP,
        object cr_dict,
        dict tmp_vars
    ):
        """
        Updates the parameter gamma (persistence) by running the EM Algorithm.

        The equations for this parameter are considerably more complex than for
        parameters alpha and sigma.
        """
        cdef:
            Factor factor
            int i = 0, j = 0, k = 0
            float ESS_0 = 0, ESS_1 = 0, ESS_denominator=0

        factor = Factor(r, last_r, doc_data['doc'], query, doc_data['click'],
                        doc_data['purchase'], self.dbn_params, cr_dict[doc_data['doc']],
                        e_r_array_given_CP, cp_array_given_e)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ESS_denominator += factor.compute_factor(i, j, k)

        ESS_0 = factor.compute_factor(1, 0, 0) / ESS_denominator
        ESS_1 = factor.compute_factor(1, 0, 1) / ESS_denominator

        tmp_vars['gamma'][0] += ESS_1
        tmp_vars['gamma'][1] += ESS_0 + ESS_1

    cdef void update_alpha(
        self,
        int r,
        str query,
        dict doc_data,
        array.array e_r_array,
        array.array X_r_array,
        int last_r,
        dict tmp_vars,
    ):
        """
        Updates the parameter alpha (attractiveness) by running the EM Algorithm.

        The equation for updating alpha is:

        \\alpha_{uq}^{(t+1)} = \\frac{\\sum_{s \\in S_{uq}}\\left(c_r^{(s)} +
          \\left(1 - c_r^{(s)}\\right)\\left(1 - c_{>r}^{(s)}\\right) \\cdot
          \\frac{\\left(1 - \\epsilon_r^{(t)}\\right)\\alpha_{uq}^{(t)}}{\\left(1 -
          \\epsilon_r^{(t)}X_r^{(t)} \\right)} \\right)}{|S_{uq}|}
        """
        cdef:
            float alpha
            str doc = doc_data['doc']
            bint click = <bint> doc_data['click']
        if doc not in tmp_vars:
            tmp_vars[doc] = {'alpha': [1, 2], 'sigma': [1, 2]}
        if click:
            tmp_vars[doc]['alpha'][0] += 1
        elif r > last_r:
            alpha = self.dbn_params.get(query, {}).get(doc, {}).get('alpha', random.random())
            tmp_vars[doc]['alpha'][0] += (
                (1 - e_r_array[r]) * alpha / (1 - e_r_array[r] * X_r_array[r])
            )
        tmp_vars[doc]['alpha'][1] += 1

    cdef void update_sigma(
        self,
        str query,
        int r,
        dict doc_data,
        array.array X_r_array,
        int last_r,
        dict tmp_vars,
    ):
        """
        Updates parameter sigma (satisfaction) by running the EM Algorithm.

        The equation for updating sigma is:

        \\sigma_{uq}^{(t+1)} = \\frac{\\sum_{s \\in S^{[1, 0]}}\\frac{(1 - c_r^{(t)})
          (1-p_r^{(t)})\\sigma_{uq}^{(t)}}{(1 - X_{r+1}\\cdot (1-\\alpha_{uq}^{(t)})
          \\gamma^{(t)})}}{|S^{[1, 0]}|}
        """
        cdef:
            float sigma
            float gamma
            str doc = doc_data['doc']
            bint click = <bint>doc_data['click']
            bint purchase = <bint>doc_data['purchase']
        if not click or purchase:
            return
        if r == last_r:
            sigma = self.dbn_params.get(query, {}).get(doc, {}).get('sigma', random.random())
            gamma = self.dbn_params.get('gamma', random.random())
            tmp_vars[doc]['sigma'][0] += (
                sigma / (1 - (X_r_array[r + 1] * (1 - sigma) * gamma))
            )
        tmp_vars[doc]['sigma'][1] += 1

    cdef array.array build_X_r_array(self, list session, str query):
        """
        X_r extends for the probability that a click will be observed at position r or
        greater given that the document at position r was examined, that is, E_r=1.

        DBN model considers that X_r for the last sku in the search result page is zero.

        This information is used to build from last to first each probability X_r.

        The equation implemented here is:

        X{_r} = P(C_{\\geq r} \\mid E_r=1) &=
          &= \\alpha_{uq} + (1 - \\alpha_{uq})\\gamma X_{r+1}

        Args
        ----
          session: List[Dict[str, Any]]
              List of documents and interactions user had for current session.
          query: str
          dbn_params: multiprocessing.ProxyDict
              Dictionary containing updated parameters for \\alpha, \\sigma and \\gamma.

        Returns
        -------
          X_r_array: array.array
              Array containing probabilities of observing clicks at position r or
              greater, given that current position was examined (E_r=1).
        """
        cdef:
            int r
            dict doc_data
            str doc
            float alpha
            float gamma
            int length = len(session)
            float X_r_previous
            float tmp
            array.array X_r_array = array.array('f', [0] * (length + 1))

        for r in range(length - 1, -1, -1):
            doc_data = session[r]
            doc = doc_data['doc']
            alpha = self.dbn_params.get(query, {}).get(doc, {}).get('alpha', random.random())
            gamma = self.dbn_params.get('gamma', random.random())
            X_r_forward = X_r_array[r + 1]
            tmp = alpha + (1 - alpha) * gamma * X_r_forward
            X_r_array[r] = tmp
        return X_r_array

    cdef array.array build_e_r_array_given_CP(self, list session, str query):
        """
        Computes the probability that a given document was examined given the array of
        previous clicks and purchases. Mathematically: P(E_r = 1 | C_{<r}, P_{<r})

        Args
        ----
          session: List[Dict[str, Any]]
              List of documents and interactions user had for current session.
          query: str
          dbn_params: multiprocessing.ProxyDict
              Dictionary containing updated parameters for \\alpha, \\sigma and \\gamma.

        Returns
        -------
          e_r_array_given_CP: array.array
              Probability that document at position r was examined (E_r=1)
        """
        cdef:
            int r
            str doc
            float alpha
            float sigma
            float gamma
            bint click
            bint purchase
            int length = len(session)
            array.array tmp
            array.array e_r_array_given_CP = array.array('f', [1])

        for r in range(length):
            doc_data = session[r]
            doc = doc_data['doc']
            click = <bint>doc_data['click']
            purchase = <bint>doc_data['purchase']
            alpha = self.dbn_params.get(query, {}).get(doc, {}).get('alpha', random.random())
            sigma = self.dbn_params.get(query, {}).get(doc, {}).get('sigma', random.random())
            gamma = self.dbn_params.get('gamma', random.random())
            if purchase:
                tmp = array.array('f', [0] * (length - r))
                array.extend(e_r_array_given_CP, tmp)
                return e_r_array_given_CP
            if click:
                tmp = array.array('f', [(1 - sigma) * gamma])
                array.extend(e_r_array_given_CP, tmp)
            else:
                tmp = array.array(
                    'f',
                    [(gamma * (1 - alpha) * e_r_array_given_CP[r]) /
                      (1 - alpha * e_r_array_given_CP[r])]
                )
                array.extend(e_r_array_given_CP, tmp)
        return e_r_array_given_CP

    cdef array.array build_CP_array_given_e(self, list session, str query, object cr_dict):
        """
        Computes the probability that Clicks and Purchases will be observed at positions
        greater than r given that position at r+1 was examined. Mathematically:

        P(C_{>r}, P_{>r} | E_{r+1})

        Args
        ----
          session: List[Dict[str, Any]]
              List of documents and interactions user had for current session.
          query: str
          dbn_params: multiprocessing.ProxyDict
              Dictionary containing updated parameters for \\alpha, \\sigma and \\gamma.
          cr_dict: Dict[str, float]
            Conversion Rate of documents for current query.

        Returns
        -------
          cp_array_given_e: array.array
              Probability of observing Clicks and Purchases at positions greater than
              r given that position r+1 was examined.
        """
        cdef:
            int r
            array.array e_r_array_given_CP
            array.array tmp_cp_p
            array.array cp_array_given_e = array.array('f', [])

        for r in range(len(session) - 1):
            e_r_array_given_CP = self.build_e_r_array_given_CP(
                session[r + 1:],
                query,
            )
            tmp_cp_p = array.array('f', [self.compute_cp_p(
                session[r + 1:],
                query,
                e_r_array_given_CP,
                cr_dict
            )])
            array.extend(cp_array_given_e, tmp_cp_p)
        return cp_array_given_e

    cdef float compute_cp_p(self, list session, str query,
                                array.array e_r_array_given_CP, object cr_dict):
        """
        Helper function that computes the probability of observing Clicks and Purchases
        at positions greater than r given that position r + 1 was examined.

        Mathematically:

        P(C_{>= r+1}, P_{>= r+1} | E_{r+1})

        Args
        ----
          session: List[Dict[str, Any]]
              List of documents and interactions user had for current session.
          query: str
          dbn_params: multiprocessing.ProxyDict
              Dictionary containing updated parameters for \\alpha, \\sigma and \\gamma.
          cr_dict: Dict[str, float]
              Conversion Rate of documents for current query.
          e_r_array_given_CP: array.array
              Probability of document being examined at position r given Clicks and
              Purchases observed before r.

        Returns
        -------
          cp_p: float
              Computes the probability of observing Clicks and Purchases at positions
              greater than r given that r + 1 was examined.
        """
        cdef:
            int r
            float cp_p
            array.array tmp_cp_p = array.array('f', [])
            dict doc_data
            str doc
            bint click
            bint purchase
            float alpha
            array.array tmp

        for r in range(len(session)):
            doc_data = session[r]
            doc = doc_data['doc']
            click = <bint>doc_data['click']
            purchase = <bint>doc_data['purchase']
            alpha = self.dbn_params.get(query, {}).get(doc, {}).get('alpha', random.random())
            if purchase:
                tmp = array.array('f', [cr_dict[doc] * alpha * e_r_array_given_CP[r]])
                array.extend(tmp_cp_p, tmp)
                continue
            if click:
                tmp = array.array(
                    'f',
                    [(1 - cr_dict[doc]) * alpha * e_r_array_given_CP[r]]
                )
                array.extend(tmp_cp_p, tmp)
            else:
                tmp = array.array('f', [1 - alpha * e_r_array_given_CP[r]])
                array.extend(tmp_cp_p, tmp)
        cp_p = np.prod(tmp_cp_p)
        return cp_p

    cdef void compute_cr(self, str query, list sessions, object cr_dict):
        """
        pyClickModels can also consider data related to purchases events. This method
        computes the conversion rate (cr) that each document had on each observed
        query context.

        Args
        ----
          query: str
              Query context.
          sessions: List[Dict[str, List[Dict[str, str]]]]
              List of session ids where each session contains all documents a given user
              interacted with along with clicks and purchases
          cr_dict: multiprocessing.Manager
              ProxyDict with queries as keys and values as another dict whose keys are
              documents and values are the conversion rate.
        """
        cdef dict data
        cdef array.array storage
        cdef dict session
        cdef str doc
        cdef int purchase_flg

        if query in cr_dict:
            # This means current query has already been analyzed
            return
        tmp_data = {}

        for session in sessions:
            for doc_data in list(session.values())[0]:
                doc = doc_data['doc']
                if doc not in tmp_data:
                    tmp_data[doc] = array.array('I', [0, 0])
                purchase_flg = <bint>doc_data['purchase']
                tmp_data[doc][0] += purchase_flg
                tmp_data[doc][1] += 1
        cr_dict[query] = {
            doc: storage[0] / storage[1] for doc, storage in tmp_data.items()
        }

    cdef str get_search_context_string(self, dict search_keys):
        """
        In pyClickModels, the input data can contain not only the search the user
        inserted but also more information that describes the context of the search,
        such as the region of user, their favorite brands or average purchasing price
        and so on. The computation of Judgments happens, therefore, not only on top of
        the search term but also on the context at which the search was made.

        This method combines all those keys together so the optimization happens on
        a single string as the final query.

        Args
        ----
          search_keys: dict
              Context at which search happened, for instance:
              `{"search_term": "query", "region": "northeast",
                "favorite_brand": "brand"}`

        Returns
        -------
          final_query: str
              string with sorted values joined by the `_` character.
        """
        cdef list keys = [str(e) for e in search_keys.values()]
        cdef str final_query = '_'.join([str(e) for e in sorted(keys)])
        return final_query

    cpdef void fit(self, str input_folder, int processes=0, int iters=30):
        """
        Reads through data of queries and customers sessions to find appropriate values
        of `\\alpha_{uq}` (attractiveness), `\\sigma_{uq}` (satisfaction) and `\\gama`
        (persistence) where `u` represents the document and `q` the input query.

        Args
        ----
          input_folder: str
              Path where gzipped files of queries and sessions are located. Each file
              is read in parallel for performance reasons. Here's an example of the
              expected input data on each compressed file:

              `{
                   "search_keys": {
                       "search_term": "query",
                      "key0": "value0"
                   },
                   "judgment_keys": [
                       {
                           "session ID 1": [
                               {"click": 0, "purchase": 0, "doc": "document0"}
                           ]
                       }
                   ]
               }`

              `search_keys` contains all keys that describe and are associated to the
              search term as inserted by the user. `key0` for instance could mean any
              further description of context such as the region of user, their
              preferences among many possibilities.
          processes: int
              How many cores to use for parallel processing.
          iters: int
              Total iterations the fitting method should run in the optimization
              process. The implemented algorithm is Expectation-Maximization which means
              the more iterations the more guaranteed it is values will converge.
        """
        cpu_cores = cpu_count()
        if processes == 0 or processes > cpu_cores:
            processes = cpu_cores

        files = glob.glob(os.path.join(input_folder, 'jud*'))

        self.dbn_params = {}
        cr_dict = {}
        self.dbn_params['gamma'] = random.random()

        for _ in range(iters):
            for file_ in files:
                self.run_EM(file_, cr_dict)
