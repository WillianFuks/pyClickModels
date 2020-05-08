from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from pyClickModels.jsonc cimport *


cdef class DBNModel:
    cdef:
        string get_search_context_string(self, lh_table *tbl)
        void compute_cr(self, string query, json_object *sessions, unordered_map[string, unordered_map[string, float]] *cr_dict)
