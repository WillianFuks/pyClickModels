from libcpp.string cimport string
from pyClickModels.jsonc cimport *


cdef class DBNModel:
    cdef:
        string get_search_context_string(self, lh_table *tbl)
