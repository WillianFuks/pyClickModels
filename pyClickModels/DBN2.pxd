from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from pyClickModels.jsonc cimport *


cdef class DBNModel:
    cdef:
        unordered_map[string, unordered_map[string, float]] alpha_params
        unordered_map[string, unordered_map[string, float]] sigma_params
        float gamma_param
        string get_search_context_string(self, lh_table *tbl)
        void compute_cr(self, string *query, json_object *sessions, unordered_map[string, unordered_map[string, float]] *cr_dict)
        float *get_param(self, string param, string *query, string *doc, int seed=*)
        vector[float] build_e_r_vector(self, json_object *session, string *query, unordered_map[string, float] *cr_dict)
        vector[float] build_X_r_vector(self, json_object *session, string *query)
        vector[float] build_e_r_vector_given_CP(self, json_object *session, string *query)
