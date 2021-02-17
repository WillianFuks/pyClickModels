from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from pyClickModels.jsonc cimport *


cdef class DBNModel:
    cpdef void fit(self, str input_folder, int iters=*)
    cpdef void export_judgments(self, str output, str format_=*)
    cdef:
        float gamma_param
        unordered_map[string, unordered_map[string, float]] alpha_params
        unordered_map[string, unordered_map[string, float]] sigma_params
        string get_search_context_string(self, lh_table *tbl)
        void compute_cr(self, string *query, json_object *sessions, unordered_map[string, unordered_map[string, float]] *cr_dict)
        float *get_param(self, string param, string *query=*, string *doc=*)
        vector[float] build_e_r_vector(self, json_object *clickstream, string *query, unordered_map[string, float] *cr_dict)
        vector[float] build_X_r_vector(self, json_object *clisktream, string *query)
        vector[float] build_e_r_vector_given_CP(self, json_object *clickstream, unsigned int idx, string *query)
        float compute_cp_p(self, json_object *clickstream, unsigned int idx, string *query, vector[float] *e_r_array_given_CP, unordered_map[string, float] *cr_dict)
        vector[float] build_CP_vector_given_e(self, json_object *session, string *query, unordered_map[string, float] *cr_dict)
        int get_last_r(self, json_object *clickstream, const char *event=*)
        void update_tmp_alpha(self, int r, string *query, json_object *doc_data, vector[float] *e_r_vector, vector[float] *X_r_vector, int last_r, unordered_map[string, vector[float]] *tmp_alpha_param)
        void update_tmp_sigma(self, string *query, int r, json_object *doc_data, vector[float] *X_r_vector, int last_r, unordered_map[string, vector[float]] *tmp_sigma_param)
        void update_tmp_gamma(self, int r, int last_r, json_object *doc_data, string *query, vector[float] *cp_vector_given_e, vector[float] *e_r_vector_given_CP, unordered_map[string, float] *cr_dict, vector[float] *tmp_gamma_param)
        void update_alpha_param(self, string *query, unordered_map[string, vector[float]] *tmp_alpha_param)
        void update_sigma_param(self, string *query, unordered_map[string, vector[float]] *tmp_sigma_param)
        void update_gamma_param(self, vector[float] *tmp_gamma_param)
        void update_tmp_params(self, json_object *session, unordered_map[string, vector[float]] *tmp_alpha_param, unordered_map[string, vector[float]] *tmp_sigma_param, vector[float] *tmp_gamma_param, string *query, unordered_map[string, float] *cr_dict)
        void restart_tmp_params(self, unordered_map[string, vector[float]] *tmp_alpha_param, unordered_map[string, vector[float]] *tmp_sigma_param, vector[float] *tmp_gamma_param)

cdef class Factor:
    cdef:
        unsigned int r
        unsigned int last_r
        bint click
        bint purchase
        float alpha
        float sigma
        float gamma
        float cr
        vector[float] *e_r_vector_given_CP
        vector[float] *cp_vector_given_e
        float compute_factor(self, bint x, bint y, bint z)
        cinit(self, unsigned int r, unsigned int last_r, bint click, bint purchase, float alpha, float sigma, float gamma, float cr, vector[float] *e_r_vector_given_CP, vector[float] *cp_vector_given_e)
