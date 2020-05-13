from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time as ctime
from cython.operator cimport dereference, postincrement
from pyClickModels.jsonc cimport (json_object, json_tokener_parse,
                                  json_object_object_get_ex, json_object_get_string,
                                  lh_table, lh_entry, json_object_array_length,
                                  json_object_array_get_idx, json_object_get_int)


cdef class DBNModel():
    def __cinit__(self):
        self.gamma_param = -1

    cdef float *get_param(self, string param, string *query, string *doc,
                          int seed=0):
        """
        Gets the value of a specific parameter (can be either 'alpha', 'sigma' or
        'gamma'in DBN setup) for specific query and document. If no such key exists,
        creates it with initial random value.

        Args
        ----
          param: const char *
              Either 'alpha' or 'sigma'
          query: string
          seed: int
              Seed to set for generating random numbers. If 0 (zero) then builds it
              based on the `time` script from Cython Includes libc.
        """
        cdef unordered_map[string, unordered_map[string, float]] *tmp

        if not seed:
            seed = ctime(NULL)
        srand(seed)

        if param == b'gamma':
            # if gamma=-1 then it hasn't been initialized yet
            if self.gamma_param == -1:
                self.gamma_param = rand() / RAND_MAX
            return &self.gamma_param
        elif param == b'alpha':
            tmp = &self.alpha_params
        else:
            # param = b'sigma':
            tmp = &self.sigma_params

        # query not in map
        if tmp[0].find(query[0]) == tmp[0].end():
            # using c rand function as it's ~ 15 - 30 times faster than Python's random
            tmp[0][query[0]][doc[0]] = rand() / RAND_MAX
        # query is in map but document is not
        elif tmp[0][query[0]].find(doc[0]) == tmp[0][query[0]].end():
            tmp[0][query[0]][doc[0]] = rand() / RAND_MAX

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
            # we have a base vale for the input query as expressed by its complete
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
          cr_dict: multiprocessing.Manager
              ProxyDict with queries as keys and values as another dict whose keys are
              documents and values are the conversion rate.
        """
        # If query is already available on cr_dict then it's not required to be
        # processed again.
        if cr_dict.find(query[0]) != cr_dict.end():
            return

        cdef:
            size_t nsessions = json_object_array_length(sessions)
            size_t nclicks
            json_object *jso_session
            json_object *clickstream
            json_object *jso_click
            json_object *tmp_jso
            string doc
            bint click
            bint purchase
            unsigned int i
            vector[int] vec
            unordered_map[string, vector[int]] tmp_cr
            unordered_map[string, vector[int]].iterator it
            float cr

        for i in range(nsessions):
            jso_session = json_object_array_get_idx(sessions, i)
            json_object_object_get_ex(jso_session, b'session', &clickstream)

            nclicks = json_object_array_length(clickstream)

            for j in range(nclicks):
                jso_click = json_object_array_get_idx(clickstream, i)

                json_object_object_get_ex(jso_click, b'doc', &tmp_jso)
                doc = <string>json_object_get_string(tmp_jso)

                json_object_object_get_ex(jso_click, b'click', &tmp_jso)
                click = <bint>json_object_get_int(tmp_jso)

                json_object_object_get_ex(jso_click, b'purchase', &tmp_jso)
                purchase = <bint>json_object_get_int(tmp_jso)

                # First time seeing the document. Prepare a mapping to store total
                # purchases and total times the document appeared on a given query
                # across all sessions.
                if tmp_cr.find(doc) == tmp_cr.end():
                    tmp_cr[doc] = vector[int](2)

                if purchase:
                    tmp_cr[doc][0] += 1

                tmp_cr[doc][1] += 1

        it = tmp_cr.begin()
        while(it != tmp_cr.end()):
            cr = dereference(it).second[0] / dereference(it).second[1]
            dereference(cr_dict)[query[0]][dereference(it).first] = cr
            postincrement(it)

    cdef vector[float] build_e_r_vector(
        self,
        json_object *session,
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
          session: json_object *
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
            size_t total_docs = json_object_array_length(session)
            string doc
            unsigned int r
            json_object *tmp
            float *alpha
            float *beta
            float *gamma
            float cr
            float e_r_next
            # we add +1 to total_docs to compute P(E_{r+1})
            vector[float] e_r_vector = vector[float](total_docs + 1)

        # Probability of Examination at r=0 (first document in search page results)
        # is always 100%
        e_r_vector[0] = 1

        # we compute P(E_{r+1}) so we add +1 to the total docs
        for r in range(1, total_docs + 1):
            json_object_object_get_ex(
                json_object_array_get_idx(session, r - 1),
                b'doc',
                &tmp
            )
            doc = json_object_get_string(tmp)
            alpha = self.get_param(b'alpha', query, &doc)
            sigma = self.get_param(b'sigma', query, &doc)
            gamma = self.get_param(b'gamma', query, &doc)
            cr = dereference(cr_dict)[doc]

            e_r_next = (e_r_vector[r - 1] * gamma[0] * ((1 - sigma[0]) * (1 - cr) *
                        alpha[0] + (1 - alpha[0])))
            e_r_vector[r] = e_r_next
        return e_r_vector

    cdef vector[float] build_X_r_vector(self, json_object *session, string *query):
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
          session: *json_object
              Session clickstream (clicks and purchases)
          query: *string
        """
        cdef:
            size_t total_docs = json_object_array_length(session)
            unsigned int r
            string doc
            # we add one to the length because of the zero value added
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

        for r in range(total_docs - 1, -1, -1):
            json_object_object_get_ex(
                json_object_array_get_idx(session, r),
                b'doc',
                &tmp
            )
            doc = json_object_get_string(tmp)
            alpha = self.get_param(b'alpha', query, &doc)
            sigma = self.get_param(b'sigma', query, &doc)
            gamma = self.get_param(b'gamma', query, &doc)

            X_r_1 = X_r_vector[r + 1]
            X_r = alpha[0] + (1 - alpha[0]) * gamma[0] * X_r_1
            X_r_vector[r]  = X_r
        return X_r_vector

    cdef vector[float] build_e_r_vector_given_CP(self, json_object *session,
                                                 string *query):
        """
        Computes the probability that a given document was examined given the array of
        previous clicks and purchases.

        Mathematically: P(E_r = 1 | C_{<r}, P_{<r})

        Args
        ----
          session: *json_object
              Clickstream of user session
          query: *string

        Returns
        -------
          e_r_vector_given_CP: vector[float]
              Probability that document at position r was examined (E_r=1)
        """
        cdef:
            size_t total_docs = json_object_array_length(session)
            unsigned int r
            string doc
            float *alpha
            float *beta
            float *gamma
            bint click
            bint purchase
            json_object *tmp
            # position r + 1 will be required later on so we add +1 in computation
            vector[float] e_r_vector_given_CP = vector[float](total_docs + 1, 0.0)

        # First document has 100% of being Examined regardless of clicks or purchases.
        e_r_vector_given_CP[0] = 1

        for r in range(total_docs):
            json_object_object_get_ex(
                json_object_array_get_idx(session, r),
                b'doc',
                &tmp
            )
            doc = json_object_get_string(tmp)

            json_object_object_get_ex(
                json_object_array_get_idx(session, r),
                b'click',
                &tmp
            )
            click = <bint>json_object_get_int(tmp)

            json_object_object_get_ex(
                json_object_array_get_idx(session, r),
                b'purchase',
                &tmp
            )
            purchase = <bint>json_object_get_int(tmp)

            alpha = self.get_param(b'alpha', query, &doc)
            sigma = self.get_param(b'sigma', query, &doc)
            gamma = self.get_param(b'gamma', query, &doc)

            if purchase:
                return e_r_vector_given_CP
            elif click:
                e_r_vector_given_CP[r + 1] = (1 - sigma[0]) * gamma[0]
            else:
                e_r_vector_given_CP[r + 1] = (
                    (gamma[0] * (1 - alpha[0]) * e_r_vector_given_CP[r]) /
                    (1 - alpha[0] * e_r_vector_given_CP[r])
                )
        return e_r_vector_given_CP
