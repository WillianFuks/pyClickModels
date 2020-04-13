from cpython cimport array
import array


cdef class DBNModel:

    cpdef void fit(self, str input_folder, int processes=*, int iters=*)

    cdef:
        dbn_params

        str get_search_context_string(self, dict search_keys)
        void compute_cr(self, str query, list sessions, object cr_dict)
        void update_params(self, dict session, dict tmp_vars, str query, object dbn_params, object cr_dict)
        array.array build_e_r_array(self, list session, str query, object dbn_params, object cr_dict)
        array.array build_X_r_array(self, list session, str query, object dbn_params)
        array.array build_e_r_array_given_CP(self, list session, str query, object dbn_params)
        float compute_cp_p(self, list session, str query, object dbn_params, array.array e_r_array_given_CP, object cr_dict)
        array.array build_CP_array_given_e(self, list session, str query, object dbn_params, object cr_dict)
        int get_last_r(self, list session, str event=*)
        void update_alpha(self, int r, str query, dict doc_data, array.array e_r_array, array.array X_r_array, int last_r, dict tmp_vars, object dbn_params)
        void update_sigma(self, str query, int r, dict doc_data, array.array X_r_array, int last_r, dict tmp_vars, object dbn_params)
        void update_gamma(self, int r, int last_r, dict doc_data, str query, object dbn_params, array.array cp_array_given_e, array.array e_r_array_given_CP, object cr_dict, dict tmp_vars)
        void update_dbn_params(self, str query, object dbn_params, dict tmp_vars)

cdef class Factor:
    cdef:
        int r, last_r
        float alpha, sigma, gamma, cr
        str doc, query
        bint click, purchase
        object dbn_params
        array.array e_r_array_given_CP, cp_array_given_e
        float compute_factor(self, bint x, bint y, bint z)
