# cython: linetrace=True

import os
from glob import glob
import gzip
import time
import ujson
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time as ctime
from cython.operator cimport dereference, postincrement
from pyClickModels.jsonc cimport(json_object, json_tokener_parse,
                                 json_object_object_get_ex, json_object_get_string,
                                 lh_table, lh_entry, json_object_array_length,
                                 json_object_array_get_idx, json_object_get_int,
                                 json_object_put)


# Start by setting the seed for the random values required for initalizing the DBN
# parameters.
SEED = ctime(NULL)
srand(SEED)


cdef class Factor:
    """
    Helper class to implement the Factor component as discussed in:

    https://clickmodels.weebly.com/uploads/5/2/2/5/52257029/mc2015-clickmodels.pdf

    page 37 equation 4.43

    Args
    ----
      r: int
          Rank position in search results.
      last_r: int
          Last observed click or purchase from search results.
      click: bint
      purchase: bint
      alpha: float
          Updated values of alpha.
      sigma: float
          Updated values of sigma.
      gamma: float
          Updated value of gamma
      cr: float
          Conversion Rate of current document in session.
      vector[float] e_r_vector_given_CP*
          Probability that document at position r was examined (E_r=1) given clicks
          and purchases.
      vector[float] cp_vector_given_e*
          Probability of observing Clicks and Purchases at positions greater than
          r given that position r + 1 was examined.
    """
    # Use cinit instead of __cinit__ so to send pointers as input.
    cdef cinit(
        self,
        unsigned int r,
        unsigned int last_r,
        bint click,
        bint purchase,
        float alpha,
        float sigma,
        float gamma,
        float cr,
        vector[float] *e_r_vector_given_CP,
        vector[float] *cp_vector_given_e
    ):
        self.r = r
        self.last_r = last_r
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.click = <bint>click
        self.purchase = <bint>purchase
        self.cr = cr
        self.e_r_vector_given_CP = e_r_vector_given_CP
        self.cp_vector_given_e = cp_vector_given_e

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
        # Compute P(C_{>r},P_{>r} | E_{r+1})
        if not z:
            if self.last_r >= self.r + 1:
                return 0
        else:
            if self.r < self.cp_vector_given_e[0].size():
                result *= self.cp_vector_given_e[0][self.r]
        # P(E_r=x | C<r, P<r)
        result *= (self.e_r_vector_given_CP[0][self.r] if x else
                   1 - self.e_r_vector_given_CP[0][self.r])
        return result


cpdef DBN():  # pragma: no cover
    return DBNModel()


cdef class DBNModel():
    def __cinit__(self):
        self.gamma_param = -1

    cdef float *get_param(self, string param, string *query=NULL, string *doc=NULL):
        """
        Gets the value of a specific parameter (can be either 'alpha', 'sigma' or
        'gamma'in DBN setup) for specific query and document. If no such key exists,
        creates it with initial random value.

        Args
        ----
          param: const char *
              Either 'alpha' or 'sigma'.
          query: string*
          seed: int
              Seed to set for generating random numbers. If 0 (zero) then builds it
              based on the `time` script from Cython Includes libc.
        """
        cdef:
            unordered_map[string, unordered_map[string, float]] *tmp

        if param == b'gamma':
            # if gamma=-1 then it hasn't been initialized yet
            if self.gamma_param == -1:
                self.gamma_param = <float>rand() / RAND_MAX
            return &self.gamma_param
        elif param == b'alpha':
            tmp = &self.alpha_params
        else:
            # param = b'sigma':
            tmp = &self.sigma_params

        # query not in map
        if tmp[0].find(query[0]) == tmp[0].end():
            # using c rand function as it's ~ 15 - 30 times faster than Python's random
            tmp[0][query[0]][doc[0]] = <float>rand() / RAND_MAX
        # query is in map but document is not
        elif tmp[0][query[0]].find(doc[0]) == tmp[0][query[0]].end():
            tmp[0][query[0]][doc[0]] = <float>rand() / RAND_MAX

        return &tmp[0][query[0]][doc[0]]

    cdef string get_search_context_string(self, lh_table *tbl):
        """
        In pyClickModels, the input data can contain not only the search the user
        inserted but also more information that describes the context of the search,
        such as the region of user, their favorite brands or average purchasing price
        and so on.

        The computation of Judgments happens, therefore, not only on top of the search
        term but also on the context at which the search was made.

        This method combines all those keys together so the optimization happens on
        a single string as the final query.

        Args
        ----
          search_keys: lh_table
              Context at which search happened, expressed in JSON. Example:
              `{"search_term": "query", "region": "northeast", "avg_ticket": 20}`

        Returns
        -------
          final_query: str
              string with sorted values joined by the `_` character.
        """
        cdef:
            string result
            char *k
            json_object *v
            lh_entry *entry = tbl.head

        k = <char *>entry.k
        v = <json_object *>entry.v
        # CPython now optimizes `+` operations. It's expected Cython will have the same
        # compilation rules.
        result = string(k) + string(b':') + string(json_object_get_string(v))

        entry = entry.next
        while entry:
            k = <char *>entry.k
            v = <json_object *>entry.v
            # Stores keys and values separated by ":" and then by "|". This is done so
            # there's a base vale for the input query as expressed by its complete
            # context (context here means possible keys that discriminate the search
            # such as the region of user, favorite brand, average ticket and so on.
            result = (
                result + string(b'|') + string(k) + string(b':') +
                string(json_object_get_string(v))
            )
            entry = entry.next
        return result

    cdef void compute_cr(self, string *query, json_object *sessions,
                         unordered_map[string, unordered_map[string, float]] *cr_dict):
        """
        pyClickModels can also consider data related to purchases events. This method
        computes the conversion rate (cr) that each document had on each observed
        query context.

        Args
        ----
          query: *string
          sessions: *json_object
              List of session ids where each session contains all documents a given user
              interacted with along with clicks and purchases
          cr_dict: unordered_map[string, float]]
              Map of documents and their respective conversion rates for each specific
              query.
        """
        # If query is already available on cr_dict then it's not required to be
        # processed again.
        if cr_dict[0].find(query[0]) != cr_dict[0].end():
            return

        cdef:
            size_t nsessions = json_object_array_length(sessions)
            size_t nclicks
            json_object *jso_session
            json_object *clickstream
            json_object *doc_data
            json_object *tmp_jso
            string doc
            bint click
            bint purchase
            unsigned int i
            unsigned int j
            vector[int] vec
            unordered_map[string, vector[int]] tmp_cr
            unordered_map[string, vector[int]].iterator it
            float cr

        for i in range(nsessions):
            jso_session = json_object_array_get_idx(sessions, i)
            json_object_object_get_ex(jso_session, b'session', &clickstream)

            nclicks = json_object_array_length(clickstream)

            for j in range(nclicks):
                doc_data = json_object_array_get_idx(clickstream, j)

                json_object_object_get_ex(doc_data, b'doc', &tmp_jso)
                doc = <string>json_object_get_string(tmp_jso)

                json_object_object_get_ex(doc_data, b'click', &tmp_jso)
                click = <bint>json_object_get_int(tmp_jso)

                json_object_object_get_ex(doc_data, b'purchase', &tmp_jso)
                purchase = <bint>json_object_get_int(tmp_jso)

                # First time seeing the document. Prepare a mapping to store total
                # purchases and total times the document appeared on a given query
                # across all sessions.
                if tmp_cr.find(doc) == tmp_cr.end():
                    tmp_cr[doc] = vector[int](2)
                    tmp_cr[doc][0] = 0
                    tmp_cr[doc][1] = 0

                if purchase:
                    tmp_cr[doc][0] += 1

                tmp_cr[doc][1] += 1

        it = tmp_cr.begin()
        while(it != tmp_cr.end()):
            cr = <float>dereference(it).second[0] / dereference(it).second[1]
            cr_dict[0][query[0]][dereference(it).first] = cr
            postincrement(it)

    cdef vector[float] build_e_r_vector(
        self,
        json_object *clickstream,
        string *query,
        unordered_map[string, float] *cr_dict,
    ):
        """
        Computes the probability of each document in user session being examined.

        The equation implemented is:

        $P(E_{r+1}=1) = \\epsilon_r \\gamma \\left((1 - \\alpha_{uq}) +
            (1 - \\sigma_{uq})(1 - cr_{uq})\\alpha_{uq} \\right)$

        Args
        ----
          clickstream: json_object *
              JSON obect representing the user clickstream. Example:
                  [
                      {"doc": "doc0", "click": 0, "purchase": 0},
                      {"doc": "doc1", "click": 1, "purchase": 0}
                  ]
          query: string
          cr_dict: unordered_map[string, float] *
              Conversion rates of each document for a given query. Example:
              {"doc0": 0.2, "doc1": 0.51}

        Returns
        -------
          e_r_vector: vector[float]
              vector to receive final probabilities
        """
        cdef:
            size_t total_docs = json_object_array_length(clickstream)
            string doc
            unsigned int r
            json_object *tmp
            float *alpha
            float *beta
            float *gamma
            float cr
            float e_r_next
            # Add +1 to total_docs to compute P(E_{r+1})
            vector[float] e_r_vector = vector[float](total_docs + 1)

        # Probability of Examination at r=0 (first document in search page results)
        # is always 100%
        e_r_vector[0] = 1

        # Compute P(E_{r+1}) so add +1 to the total docs
        for r in range(1, total_docs + 1):
            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r - 1),
                b'doc',
                &tmp
            )
            doc = json_object_get_string(tmp)
            alpha = self.get_param(b'alpha', query, &doc)
            sigma = self.get_param(b'sigma', query, &doc)
            gamma = self.get_param(b'gamma')
            cr = dereference(cr_dict)[doc]

            e_r_next = (e_r_vector[r - 1] * gamma[0] * ((1 - sigma[0]) * (1 - cr) *
                        alpha[0] + (1 - alpha[0])))
            e_r_vector[r] = e_r_next
        return e_r_vector

    cdef vector[float] build_X_r_vector(self, json_object *clickstream, string *query):
        """
        X_r is given by P(C_{\\geq r} \\mid E_r=1). It extends for the probability of
        click on any rank starting from current until last one. This vector is also
        used in the EM optimization process.

        The probability of click after the very last sku is considered zero. This
        allows to build the `X_r` vector recursively.

        The equation is:

        X{_r} = P(C_{\\geq r} \\mid E_r=1) &=
          &= \\alpha_{uq} + (1 - \\alpha_{uq})\\gamma X_{r+1}

        Args
        ----
          clickstream: *json_object
              Session clickstream (clicks and purchases)
          query: *string
        """
        cdef:
            size_t total_docs = json_object_array_length(clickstream)
            unsigned int r
            string doc
            # Add one to the length because of the zero value added for position
            # N + 1 where N is the amount of documents returned in the search page.
            vector[float] X_r_vector = vector[float](total_docs + 1)
            json_object *tmp
            float X_r_1
            float X_r
            float *alpha
            float *beta
            float *gamma

        # Probability of clicks at positions greater than the last document in results
        # page is zero.
        X_r_vector[total_docs] = 0
        gamma = self.get_param(b'gamma')

        for r in range(total_docs - 1, -1, -1):
            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r),
                b'doc',
                &tmp
            )
            doc = json_object_get_string(tmp)
            alpha = self.get_param(b'alpha', query, &doc)

            X_r_1 = X_r_vector[r + 1]
            X_r = alpha[0] + (1 - alpha[0]) * gamma[0] * X_r_1
            X_r_vector[r] = X_r
        return X_r_vector

    cdef vector[float] build_e_r_vector_given_CP(self, json_object *clickstream,
                                                 unsigned int idx, string *query):
        """
        Computes the probability that a given document was examined given the array of
        previous clicks and purchases.

        Mathematically: P(E_r = 1 | C_{<r}, P_{<r})

        This is discussed in equation (24) in the blog post:

        https://towardsdatascience.com/how-to-extract-relevance-from-clickstream-data-2a870df219fb

        Args
        ----
          clickstream: *json_object
              Clickstream of user session
          idx: unsigned int
              Index from where to start slicing sessions
          query: *string

        Returns
        -------
          e_r_vector_given_CP: vector[float]
              Probability that document at position r was examined (E_r=1) given clicks
              and purchases.
        """
        cdef:
            size_t total_docs = json_object_array_length(clickstream)
            unsigned int r
            string doc
            float *alpha
            float *beta
            float *gamma
            bint click
            bint purchase
            json_object *tmp
            # position r + 1 will be required later so add +1 in computation
            vector[float] e_r_vector_given_CP = vector[float](total_docs + 1 - idx, 0.0)

        # First document has 100% chance of being Examined regardless of clicks or
        # purchases.
        e_r_vector_given_CP[0] = 1
        gamma = self.get_param(b'gamma')

        for r in range(idx, total_docs):
            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r),
                b'doc',
                &tmp
            )
            doc = json_object_get_string(tmp)

            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r),
                b'click',
                &tmp
            )
            click = <bint>json_object_get_int(tmp)

            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r),
                b'purchase',
                &tmp
            )
            purchase = <bint>json_object_get_int(tmp)

            alpha = self.get_param(b'alpha', query, &doc)
            sigma = self.get_param(b'sigma', query, &doc)

            if purchase:
                return e_r_vector_given_CP
            elif click:
                e_r_vector_given_CP[r + 1 - idx] = (1 - sigma[0]) * gamma[0]
            else:
                e_r_vector_given_CP[r + 1 - idx] = (
                    (gamma[0] * (1 - alpha[0]) * e_r_vector_given_CP[r - idx]) /
                    (1 - alpha[0] * e_r_vector_given_CP[r - idx])
                )
        return e_r_vector_given_CP

    cdef float compute_cp_p(
        self,
        json_object *clickstream,
        unsigned int idx,
        string *query,
        vector[float] *e_r_array_given_CP,
        unordered_map[string, float] *cr_dict
    ):
        """
        Helper function that computes the probability of observing Clicks and Purchases
        at positions greater than r given that position r + 1 was examined.

        Mathematically:

        P(C_{>= r+1}, P_{>= r+1} | E_{r+1})

        Args
        ----
          session: *json_object
              Customer's clickstream.
          idx: unsigned int
              Index from where to start slicing json session
          query: *string
          cr_dict: unordered_map[string, float] *cr_dict
              Conversion Rate (CR) of documents for current query
          e_r_array_given_CP: vector[float]
              Probability of document being examined at position r given Clicks and
              Purchases observed before r.

        Returns
        -------
          cp_p: float
              Computes the probability of observing Clicks and Purchases at positions
              greater than r given that r + 1 was examined.
        """
        cdef:
            size_t total_docs = json_object_array_length(clickstream)
            unsigned int r
            string doc
            float *alpha
            bint click
            bint purchase
            json_object *tmp
            float cp_p = 1

        for r in range(idx, total_docs):
            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r),
                b'doc',
                &tmp
            )
            doc = json_object_get_string(tmp)

            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r),
                b'click',
                &tmp
            )
            click = <bint>json_object_get_int(tmp)

            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r),
                b'purchase',
                &tmp
            )
            purchase = <bint>json_object_get_int(tmp)

            alpha = self.get_param(b'alpha', query, &doc)

            # Subtract `idx` from `r` because the input `e_r_array_given_CP`
            # should always be counted from the beginning (despite the slicing in
            # sessions, this variable should still be counted as if the new session
            # is not a slice of any sort).
            if purchase:
                cp_p *= cr_dict[0][doc] * alpha[0] * e_r_array_given_CP[0][r - idx]
            elif click:
                cp_p *= (
                    (1 - cr_dict[0][doc]) * alpha[0] * e_r_array_given_CP[0][r - idx]
                )
            else:
                cp_p *= 1 - alpha[0] * e_r_array_given_CP[0][r - idx]
        return cp_p

    cdef vector[float] build_CP_vector_given_e(
        self,
        json_object *clickstream,
        string *query,
        unordered_map[string, float] *cr_dict
    ):
        """
        Computes the probability that Clicks and Purchases will be observed at positions
        greater than r given that position at r+1 was examined.

        Mathematically:

        P(C_{>r}, P_{>r} | E_{r+1})

        This is equation (25) from blog post:

        https://towardsdatascience.com/how-to-extract-relevance-from-clickstream-data-2a870df219fb

        Args
        ----
          clickstream: *json_object
              User clickstream
          query: *string
          cr_dict: *unordered_map[string, float]
            Conversion Rate (CR) of documents for current query

        Returns
        -------
          cp_vector_given_e: vector[float]
              Probability of observing Clicks and Purchases at positions greater than
              r given that position r + 1 was examined.
        """
        cdef:
            unsigned int r
            size_t total_docs = json_object_array_length(clickstream)
            vector[float] e_r_vector_given_CP
            vector[float] cp_vector_given_e = vector[float](total_docs - 1)

        # Subtract 1 as E_{r+1} is defined up to r - 1 documents
        for r in range(total_docs - 1):
            e_r_vector_given_CP = self.build_e_r_vector_given_CP(clickstream, r + 1,
                                                                 query)
            cp_vector_given_e[r] = self.compute_cp_p(clickstream, r + 1, query,
                                                     &e_r_vector_given_CP, cr_dict)
        return cp_vector_given_e

    cdef int get_last_r(self, json_object *clickstream, const char *event=b'click'):
        """
        Loops through all documents in session and find at which position the desired
        event happend. It can be either a 'click' or a 'purchase' (still, in DBN, if
        a purchase is observed then it automatically means it is the very last r
        observed).

        Args
        ----
          session: *json_object
              User clickstream
          event: const char*
              Name of desired event to track.

        Returns
        -------
          last_r: int
              Index at which the last desired event was observed.
        """
        cdef:
            unsigned int r
            size_t total_docs = json_object_array_length(clickstream)
            unsigned int idx = 0
            json_object *tmp
            bint value

        for r in range(total_docs):
            json_object_object_get_ex(
                json_object_array_get_idx(clickstream, r),
                event,
                &tmp
            )
            value = <bint>json_object_get_int(tmp)
            if value:
                idx = r
        return idx

    cdef void update_tmp_alpha(
        self,
        int r,
        string *query,
        json_object *doc_data,
        vector[float] *e_r_vector,
        vector[float] *X_r_vector,
        int last_r,
        unordered_map[string, vector[float]] *tmp_alpha_param
    ):
        """
        Updates the parameter alpha (attractiveness) by running the EM Algorithm.

        The equation for updating alpha is:

        \\alpha_{uq}^{(t+1)} = \\frac{\\sum_{s \\in S_{uq}}\\left(c_r^{(s)} +
          \\left(1 - c_r^{(s)}\\right)\\left(1 - c_{>r}^{(s)}\\right) \\cdot
          \\frac{\\left(1 - \\epsilon_r^{(t)}\\right)\\alpha_{uq}^{(t)}}{\\left(1 -
          \\epsilon_r^{(t)}X_r^{(t)} \\right)} \\right)}{|S_{uq}|}

        Args
        ----
          r: int
              Rank position.
          query: string*
          doc_data: json_object*
              JSON object describing specific document from the search results page
              in the clickstream of a specific user.
          e_r_vector: vector[float]
              Probability of Examination at position r.
          X_r_vector: vector[float]
              Probability of clicks at position greater than r given that position r
              was Examined (E=1).
          last_r: int
              Last position r where click or purchase is observed.
          tmp_alpha_param: unordered_map[string, vector[int]]
              Holds temporary data for updating the alpha parameter.
        """
        cdef:
            float *alpha
            string doc
            bint click
            json_object *tmp

        json_object_object_get_ex(doc_data, b'doc', &tmp)
        doc = json_object_get_string(tmp)

        json_object_object_get_ex(doc_data, b'click', &tmp)
        click = <bint>json_object_get_int(tmp)

        # doc not available yet.
        if tmp_alpha_param[0].find(doc) == tmp_alpha_param[0].end():
            tmp_alpha_param[0][doc] = vector[float](2)
            tmp_alpha_param[0][doc][0] = 1
            tmp_alpha_param[0][doc][1] = 2

        if click:
            tmp_alpha_param[0][doc][0] += 1
        elif r > last_r:
            alpha = self.get_param(b'alpha', query, &doc)

            tmp_alpha_param[0][doc][0] += (
                (1 - e_r_vector[0][r]) * alpha[0] /
                (1 - e_r_vector[0][r] * X_r_vector[0][r])
            )
        tmp_alpha_param[0][doc][1] += 1

    cdef void update_tmp_sigma(
        self,
        string *query,
        int r,
        json_object *doc_data,
        vector[float] *X_r_vector,
        int last_r,
        unordered_map[string, vector[float]] *tmp_sigma_param,
    ):
        """
        Updates parameter sigma (satisfaction) by running the EM Algorithm.

        The equation for updating sigma is:

        \\sigma_{uq}^{(t+1)} = \\frac{\\sum_{s \\in S^{[1, 0]}}\\frac{(1 - c_r^{(t)})
          (1-p_r^{(t)})\\sigma_{uq}^{(t)}}{(1 - X_{r+1}\\cdot (1-\\sigma_{uq}^{(t)})
          \\gamma^{(t)})}}{|S^{[1, 0]}|}

        Args
        ----
          query: string*
          r: int
              Rank position.
          doc_data: json_object*
              Clickstream data at position r.
          X_r_vector: vector[float]
              Probability of clicks at position greater than r given that position r
              was Examined (E=1).
          last_r: int
              Last position r where click or purchase is observed.
        """
        cdef:
            float *sigma
            bint click
            json_object *tmp
            string doc

        json_object_object_get_ex(doc_data, b'doc', &tmp)
        doc = json_object_get_string(tmp)

        json_object_object_get_ex(doc_data, b'click', &tmp)
        click = <bint>json_object_get_int(tmp)

        json_object_object_get_ex(doc_data, b'purchase', &tmp)
        purchase = json_object_get_int(tmp)

        # doc not available yet.
        if tmp_sigma_param[0].find(doc) == tmp_sigma_param[0].end():
            tmp_sigma_param[0][doc] = vector[float](2)
            tmp_sigma_param[0][doc][0] = 1
            tmp_sigma_param[0][doc][1] = 2

        # satisfaction is only defined for ranks where click or no purchase were
        # observed.
        if not click or purchase:
            return

        if r == last_r:
            sigma = self.get_param(b'sigma', query, &doc)
            gamma = self.get_param(b'gamma')

            tmp_sigma_param[0][doc][0] += (
                sigma[0] / (1 - (X_r_vector[0][r + 1] * (1 - sigma[0]) * gamma[0]))
            )
        tmp_sigma_param[0][doc][1] += 1

    cdef void update_tmp_gamma(
        self,
        int r,
        int last_r,
        json_object *doc_data,
        string *query,
        vector[float] *cp_vector_given_e,
        vector[float] *e_r_vector_given_CP,
        unordered_map[string, float] *cr_dict,
        vector[float] *tmp_gamma_param
    ):
        """
        Updates the parameter gamma (persistence) by running the EM Algorithm.

        The equations for this parameter are considerably more complex than for
        parameters alpha and sigma. Using the Factor extension method to help out in
        the computation.


        Args
        ----
          r: int
              Rank position.
          last_r: int
              Last rank where either click or purchase was observed.
          doc_data: json_object*
              JSON object with clickstream information of document at position r.
          query: string*
          cp_vector_given_e: vector[float]*
              Probability of observing Clicks and Purchases at positions greater than
              r given that position r + 1 was examined.
          e_r_vector_given_CP: vector[float]*
              Probability that document at position r was examined (E_r=1) given clicks
              and purchases.
          cr_dict: unordered_map[string, float]*
              Conversion Rate of documents for respective query.
          tmp_gamma_param: vector[float]*
              Temporary updates for gamma.
        """
        cdef:
            Factor factor
            bint i = 0
            bint j = 0
            bint k = 0
            float ESS_0 = 0
            float ESS_1 = 0
            float ESS_denominator = 0
            float alpha
            float sigma
            float gamma
            json_object *tmp
            string doc
            bint click
            bint purchase
            float cr

        json_object_object_get_ex(doc_data, b'doc', &tmp)
        doc = json_object_get_string(tmp)

        json_object_object_get_ex(doc_data, b'click', &tmp)
        click = json_object_get_int(tmp)

        json_object_object_get_ex(doc_data, b'purchase', &tmp)
        purchase = json_object_get_int(tmp)

        alpha = self.get_param(b'alpha', query, &doc)[0]
        sigma = self.get_param(b'sigma', query, &doc)[0]
        gamma = self.get_param(b'gamma')[0]

        cr = cr_dict[0][doc]

        factor = Factor()
        factor.cinit(
            r,
            last_r,
            click,
            purchase,
            alpha,
            sigma,
            gamma,
            cr,
            e_r_vector_given_CP,
            cp_vector_given_e
        )

        # Loop through all possible values of x, y and z, where each is an integer
        # boolean.
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ESS_denominator += factor.compute_factor(i, j, k)

        if not ESS_denominator:
            ESS_0, ESS_1 = 0, 0
        else:
            ESS_0 = factor.compute_factor(1, 0, 0) / ESS_denominator
            ESS_1 = factor.compute_factor(1, 0, 1) / ESS_denominator

        tmp_gamma_param[0][0] += ESS_1
        tmp_gamma_param[0][1] += ESS_0 + ESS_1

    cdef void update_alpha_param(
        self,
        string *query,
        unordered_map[string, vector[float]] *tmp_alpha_param,
    ):
        """
        After all sessions for a given query have been analyzed, the new values of
        alpha in `tmp_alpha_param` are copied into `alpha_params` where they'll
        be used into new optimization iterations.

        Args
        ----
          query: string*
          tmp_alpha_param: unordered_map[string, vector[float]]
              Optimized values for updating alpha
        """
        cdef:
            unordered_map[string, vector[float]].iterator it = (
                tmp_alpha_param[0].begin()
            )
            string doc
            vector[float] value

        while(it != tmp_alpha_param[0].end()):
            doc = dereference(it).first
            value = dereference(it).second
            self.alpha_params[query[0]][doc] = value[0] / value[1]
            postincrement(it)

    cdef void update_sigma_param(
        self,
        string *query,
        unordered_map[string, vector[float]] *tmp_sigma_param,
    ):
        """
        After all sessions for a given query have been analyzed, the new values of
        sigma in `tmp_sigma_param` are copied into `sigma_params` where they'll
        be used into new optimization iterations.

        Args
        ----
          query: string*
          tmp_sigma_param: unordered_map[string, vector[float]]
              Optimized values for updating sigma
        """
        cdef:
            unordered_map[string, vector[float]].iterator it = (
                tmp_sigma_param[0].begin()
            )
            string doc
            vector[float] value

        while(it != tmp_sigma_param[0].end()):
            doc = dereference(it).first
            value = dereference(it).second
            self.sigma_params[query[0]][doc] = value[0] / value[1]
            postincrement(it)

    cdef void update_gamma_param(
        self,
        vector[float] *tmp_gamma_param
    ):
        """
        After all sessions for a given query have been analyzed, the new value of
        gamma in `tmp_sigma_param` is copied into `gamma_param` where they'll
        be used into new optimization iterations.

        Args
        ----
          tmp_gamma_param: vector[float]*
              Optimized values for updating sigma
        """
        # Considered that a denominator of zero cannot happen.
        self.gamma_param = tmp_gamma_param[0][0] / tmp_gamma_param[0][1]

    cpdef void export_judgments(self, str output, str format_='NEWLINE_JSON'):
        """
        After running the fit optimization process, exports judgment results to an
        external file in accordance to the selected input `format_`. Judgments are
        computed as:

            J_{uq} = P(\\alpha_{uq}) \\cdot P(\\sigma_{uq})

        where `u` represents the document and `q` the query.

        Args
        ----
          output: str
              Filepath where to save results. If `gz` is present in `output` then
              compresses file.
          format_: str
              Sets how to write result file. Options includes:
               - NEWLINE_JSON: writes in JSON format, like:
               {'query0': {'doc0': 0.3, 'doc1': 0.2}}
               {'query1': {'doc0': 0.76, 'doc1': 0.41}}
        """
        cdef:
            unordered_map[string, unordered_map[string, float]].iterator it
            unordered_map[string, float].iterator doc_it
            string query
            string doc
            float alpha
            float sigma
            dict tmp

        file_manager = gzip.GzipFile if '.gz' in output else open

        with file_manager(output, 'wb') as f:
            it = self.alpha_params.begin()
            while(it != self.alpha_params.end()):
                query = dereference(it).first
                tmp = {}
                tmp[query] = {}
                doc_it = self.alpha_params[query].begin()
                while(doc_it != self.alpha_params[query].end()):
                    doc = dereference(doc_it).first
                    alpha = dereference(doc_it).second
                    sigma = self.sigma_params[query][doc]
                    tmp[query][doc] = alpha * sigma
                    postincrement(doc_it)
                f.write(ujson.dumps(tmp).encode() + '\n'.encode())
                postincrement(it)

    cpdef void fit(self, str input_folder, int iters=30):
        """
        Reads through data of queries and customers sessions to find appropriate values
        of `\\alpha_{uq}` (attractiveness), `\\sigma_{uq}` (satisfaction) and `\\gama`
        (persistence) where `u` represents the document and `q` the input query.

        Args
        ----
          input_folder: str
              Path where gzipped clickstream files are located. Each file. Here's an
              example of the expected input data on each compressed file:

              `{
                   "search_keys": {
                       "search_term": "query",
                      "key0": "value0"
                   },
                   "judgment_keys": [
                       {
                           "session": [
                               {"click": 0, "purchase": 0, "doc": "document0"}
                           ]
                       }
                   ]
               }`

              `search_keys` contains all keys that describe and are associated to the
              search term as inserted by the user. `key0` for instance could mean any
              further description of context such as the region of user, their
              preferences among many possibilities.
          iters: int
              Total iterations the fitting method should run in the optimization
              process. The implemented algorithm is Expectation-Maximization which means
              the more iterations there are the more guaranteed it is values will
              converge.
        """
        cdef:
            list files = glob(os.path.join(input_folder, 'jud*'))
            # row has to be bytes so Cython can interchange its value between char* and
            # bytes
            bytes row
            json_object *row_json
            json_object *search_keys
            json_object *sessions
            json_object *session
            json_object *clickstream
            lh_table *search_keys_tbl
            int c = 0
            unsigned int i = 0
            string query
            unordered_map[string, vector[float]] tmp_alpha_param
            unordered_map[string, vector[float]] tmp_sigma_param
            vector[float] tmp_gamma_param = vector[float](2)
            unordered_map[string, unordered_map[string, float]] cr_dict

        for _ in range(iters):
            print('running iteration: ', _)
            for file_ in files:
                for row in gzip.GzipFile(file_, 'rb'):
                    # Start by erasing the temporary container of the parameters as
                    # each new query requires a new computation in the EM algorithm.
                    self.restart_tmp_params(&tmp_alpha_param, &tmp_sigma_param,
                                            &tmp_gamma_param)

                    row_json = json_tokener_parse(<char*>row)

                    json_object_object_get_ex(row_json, b'search_keys', &search_keys)
                    search_keys_tbl = json_object_get_object(search_keys)

                    query = self.get_search_context_string(search_keys_tbl)
                    json_object_object_get_ex(row_json, b'judgment_keys', &sessions)
                    self.compute_cr(&query, sessions, &cr_dict)

                    for i in range(json_object_array_length(sessions)):
                        session = json_object_array_get_idx(sessions, i)
                        json_object_object_get_ex(session, b'session', &clickstream)

                        self.update_tmp_params(clickstream, &tmp_alpha_param,
                                               &tmp_sigma_param, &tmp_gamma_param,
                                               &query, &cr_dict[query])

                    self.update_alpha_param(&query, &tmp_alpha_param)
                    self.update_sigma_param(&query, &tmp_sigma_param)
                    self.update_gamma_param(&tmp_gamma_param)
                    json_object_put(row_json)

    cdef void update_tmp_params(
        self,
        json_object *clickstream,
        unordered_map[string, vector[float]] *tmp_alpha_param,
        unordered_map[string, vector[float]] *tmp_sigma_param,
        vector[float] *tmp_gamma_param,
        string *query,
        unordered_map[string, float] *cr_dict
    ):
        """
        For each session, applies the EM algorithm and save temporary results into
        the tmp input parameters.

        Args
        ----
          clickstream: json_object*
              JSON containing documents users observed on search results page and their
              interaction with each item. Example:

                `[
                      {"doc": "doc0", "click": 0, "purchase": 0},
                      {"doc": "doc1", "click": 1, "purchase": 1}
                ]`

          tmp_alpha_param: vector[float]*
              Holds temporary values for adapting each variable alpha.
          tmp_sigma_param: vector[float]*
              Holds temporary values for adapting each variable sigma.
          tmp_gamma_param: vector[float]*
              Holds temporary values for adapting gamma.
          query: string*
          cr_dict: unordered_map[string, float]*
              Conversion Rates of each document for the current query.
        """
        cdef:
            json_object *doc_data
            vector[float] e_r_vector
            vector[float] X_r_vector
            vector[float] e_r_vector_given_CP
            vector[float] cp_vector_given_e
            unsigned int last_r
            unsigned int r

        e_r_vector = self.build_e_r_vector(clickstream, query, cr_dict)
        X_r_vector = self.build_X_r_vector(clickstream, query)
        e_r_vector_given_CP = self.build_e_r_vector_given_CP(clickstream, 0, query)
        cp_vector_given_e = self.build_CP_vector_given_e(clickstream, query, cr_dict)
        # last clicked position
        last_r = self.get_last_r(clickstream)

        for r in range(json_object_array_length(clickstream)):
            doc_data = json_object_array_get_idx(clickstream, r)
            self.update_tmp_alpha(r, query, doc_data, &e_r_vector, &X_r_vector, last_r,
                                  tmp_alpha_param)
            self.update_tmp_sigma(query, r, doc_data, &X_r_vector, last_r,
                                  tmp_sigma_param)
            self.update_tmp_gamma(r, last_r, doc_data, query, &cp_vector_given_e,
                                  &e_r_vector_given_CP, cr_dict, tmp_gamma_param)

    cdef void restart_tmp_params(
        self,
        unordered_map[string, vector[float]] *tmp_alpha_param,
        unordered_map[string, vector[float]] *tmp_sigma_param,
        vector[float] *tmp_gamma_param
    ):
        """
        Re-creates temporary parameters to be used in the optimization process for each
        query and step.
        """
        tmp_alpha_param[0].erase(
            tmp_alpha_param[0].begin(),
            tmp_alpha_param[0].end()
        )
        tmp_sigma_param[0].erase(
            tmp_sigma_param[0].begin(),
            tmp_sigma_param[0].end()
        )
        tmp_gamma_param[0][0] = 1
        tmp_gamma_param[0][1] = 2
