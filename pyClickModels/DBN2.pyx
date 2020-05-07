from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from cython.operator cimport dereference, postincrement
from pyClickModels.jsonc cimport (json_object, json_tokener_parse,
                                 json_object_object_get_ex, json_object_get_string,
                                 json_object_get_object, lh_table, lh_entry,
                                 json_object_array_length, json_object_array_get_idx)

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
              Context at which search happened, for instance:
              `{"search_term": "query", "region": "northeast",
                "favorite_brand": "brand"}`

        Returns
        -------
          final_query: str
              string with sorted values joined by the `_` character.
        """
        cdef string r = b'b'
        return r
#         print(str(tbl.size))
        # cdef string result
        # cdef char *k
        # cdef json_object *v
        # cdef lh_entry *entry = tbl.head

        # k = <char *>entry.k
        # v = <json_object *>entry.v
        # # CPython now optimizes `+` operations. It's expected Cython will have the same
        # # compilation rules.
        # result = bytes(k) + b':' + bytes(json_object_get_string(v))

        # entry = entry.next
        # while entry:
            # k = <char *>entry.k
            # v = <json_object *>entry.v
            # result += b'|' + bytes(k) + b':' + bytes(json_object_get_string(v))
            # entry = entry.next
#         return result
