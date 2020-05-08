from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from cython.operator cimport dereference, postincrement
from pyClickModels.jsonc cimport (json_object, json_tokener_parse,
                                 json_object_object_get_ex, json_object_get_string,
                                 json_object_get_object, lh_table, lh_entry,
                                 json_object_array_length, json_object_array_get_idx,
                                 json_object_get_int)

cdef class DBNModel():
    cdef string get_search_context_string(self, lh_table *tbl):
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
          search_keys: lh_table
              Context at which search happened, expressed in JSON. Example:
              `{"search_term": "query", "region": "northeast", "avg_ticket": 20}`

        Returns
        -------
          final_query: str
              string with sorted values joined by the `_` character.
        """
        cdef string result
        cdef char *k
        cdef json_object *v
        cdef lh_entry *entry = tbl.head

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

    cdef void compute_cr(self, string query, json_object *sessions,
                         unordered_map[string, unordered_map[string, float]] *cr_dict):
        """
        pyClickModels can also consider data related to purchases events. This method
        computes the conversion rate (cr) that each document had on each observed
        query context.

        Args
        ----
          query: string
              Query context.
          sessions: *json_object
              List of session ids where each session contains all documents a given user
              interacted with along with clicks and purchases
          cr_dict: multiprocessing.Manager
              ProxyDict with queries as keys and values as another dict whose keys are
              documents and values are the conversion rate.
        """
        # If query is already available on cr_dict then it's not required to be
        # processed again.
        # if cr_dict.find(query) == cr_dict.end():
            # return

        cdef size_t nsessions = json_object_array_length(sessions)
        cdef size_t nclicks
        cdef json_object *jso_session
        cdef json_object *clickstream
        cdef json_object *jso_click
        cdef json_object *tmp_jso
        cdef string doc
        cdef bint click
        cdef bint purchase
        cdef unsigned int i
        cdef vector[int] vec
        cdef unordered_map[string, vector[int]] tmp_cr
        cdef unordered_map[string, vector[int]].iterator it
        cdef float cr

        print('total sessions: ', str(nsessions))

        for i in range(nsessions):
            jso_session = json_object_array_get_idx(sessions, i)
            json_object_object_get_ex(jso_session, b'session', &clickstream)

            nclicks = json_object_array_length(clickstream)
            print('total clicks in session: ', str(nclicks))

            for j in range(nclicks):
                jso_click = json_object_array_get_idx(clickstream, i)

                json_object_object_get_ex(jso_click, b'doc', &tmp_jso)
                doc = <string>json_object_get_string(tmp_jso)
                print('this is doc: ', doc)

                json_object_object_get_ex(jso_click, b'click', &tmp_jso)
                click = <bint>json_object_get_int(tmp_jso)
                print('this is click: ', click)

                json_object_object_get_ex(jso_click, b'purchase', &tmp_jso)
                purchase = <bint>json_object_get_int(tmp_jso)
                print('this is purchase: ', purchase)

                # First time seeing the document. Prepare a mapping to store total
                # purchases and total times the document appeared on a given query
                # across all sessions.
                if tmp_cr.find(doc) == tmp_cr.end():
                    tmp_cr[doc] = vector[int](2)

                if purchase:
                    tmp_cr[doc][0] += 1

                tmp_cr[doc][1] += 1

        it = tmp_cr.begin()
        print('im getting')
        while(it != tmp_cr.end()):
            cr = dereference(it).second[0] / dereference(it).second[1]
            print('this is cr: ', cr)
            dereference(cr_dict)[query][dereference(it).first] = cr
            postincrement(it)

#         cdef dict data
        # cdef array.array storage
        # cdef dict session
        # cdef str doc
        # cdef int purchase_flg

        # if query in cr_dict:
            # # This means current query has already been analyzed
            # return
        # tmp_data = {}

        # for session in sessions:
            # for doc_data in list(session.values())[0]:
                # doc = doc_data['doc']
                # if doc not in tmp_data:
                    # tmp_data[doc] = array.array('I', [0, 0])
                # purchase_flg = <bint>doc_data['purchase']
                # tmp_data[doc][0] += purchase_flg
                # tmp_data[doc][1] += 1
        # cr_dict[query] = {
            # doc: storage[0] / storage[1] for doc, storage in tmp_data.items()
        # }

