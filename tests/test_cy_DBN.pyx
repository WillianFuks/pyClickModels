import tempfile
import ujson
import gzip
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from cython.operator cimport dereference, postincrement
from pyClickModels.DBN cimport DBNModel, Factor
from pyClickModels.jsonc cimport(json_object, json_tokener_parse,
                                 json_object_get_object, lh_table, json_object_put)
from conftest import build_DBN_test_data
from numpy.testing import assert_almost_equal, assert_allclose

ctypedef unordered_map[string, unordered_map[string, float]] dbn_param


cdef string query = b'query'
cdef dbn_param alpha_params
cdef dbn_param sigma_params
cdef float gamma_param

alpha_params[query][b'doc0'] = 0.5
alpha_params[query][b'doc1'] = 0.5
alpha_params[query][b'doc2'] = 0.5

sigma_params[query][b'doc0'] = 0.5
sigma_params[query][b'doc1'] = 0.5
sigma_params[query][b'doc2'] = 0.5

gamma_param = 0.7


cdef test_fit():
    cdef:
        DBNModel model = DBNModel()
        unordered_map[string, unordered_map[string, float]].iterator it
        string query
        string doc
    gamma, params, tmp_folder = build_DBN_test_data(users=30000, docs=8, queries=2)

    # print('expected value of sigma: ', params[0][0][1])

    model.fit(tmp_folder.name, iters=10)
    # print('model gamma ', model.gamma_param)
    # print('real gamma: ', gamma)

    # it = model.alpha_params.begin()
    # while(it != model.alpha_params.end()):
    #     prints keys
    #     print(dereference(it).first)
    #     postincrement(it)

    # print(
    #     'model.alpha_params doc 0', model.alpha_params[
    #     b'search_term:0|region:north|favorite_size:L'][b'0']
    # )
    # print('params alpha ', params[0][0][0])

    # print(
    #     'model.sigma_params doc 0', model.sigma_params[
    #     b'search_term:0|region:north|favorite_size:L'][b'0']
    # )
    # print('params sigma ', params[0][0][1])

    assert_allclose(model.gamma_param, gamma, atol=.1)
    assert_allclose(
        model.alpha_params[b'search_term:0|region:north|favorite_size:L'][b'0'],
        params[0][0][0],
        atol=.15
    )
    assert_allclose(
        model.sigma_params[b'search_term:0|region:north|favorite_size:L'][b'0'],
        params[0][0][1],
        atol=.15
    )


cdef bint test_get_search_context_string():
    cdef:
        DBNModel model = DBNModel()
        json_object *search_keys = json_tokener_parse(b"{'search_term': 'query'}")
        lh_table *tbl = json_object_get_object(search_keys)
        string result = model.get_search_context_string(tbl)
        string expected = b'search_term:query'
    assert result == expected

    search_keys = json_tokener_parse(
        b"{'search_term': 'query', 'key0': 'value0', 'key1': 'value1'}"
    )

    tbl = json_object_get_object(search_keys)
    result = model.get_search_context_string(tbl)
    assert result == b'search_term:query|key0:value0|key1:value1'

    json_object_put(search_keys)


cdef test_compute_cr():
    cdef:
        DBNModel model = DBNModel()
        string query = b'query'
        # cr_dict is like: {'query_term': {'doc0': 0.2, 'doc1: 0}}
        unordered_map[string, unordered_map[string, float]] cr_dict
        unordered_map[string, unordered_map[string, float]] expected
        const char *sessions = b"""
        [
            {
                'session': [
                    {"doc": "doc0", "click": 0, "purchase": 0},
                    {"doc": "doc1", "click": 1, "purchase": 0},
                    {"doc": "doc2", "click": 1, "purchase": 1}
                ]
            },
            {
                'session': [
                    {"doc": "doc0", "click": 0, "purchase": 0},
                    {"doc": "doc1", "click": 1, "purchase": 0},
                ]
            },
        ]
        """
        json_object *jso_sessions = json_tokener_parse(sessions)

    expected[query][b'doc0'] = <float>0
    expected[query][b'doc1'] = <float>0
    expected[query][b'doc2'] = <float>1

    model.compute_cr(&query, jso_sessions, &cr_dict)

    assert expected == cr_dict

    # test if query is already available in cr_dict
    jso_sessions = json_tokener_parse(<const char *>'')
    model.compute_cr(&query, jso_sessions, &cr_dict)
    assert expected == cr_dict

    json_object_put(jso_sessions)


cdef test_get_param():
    cdef:
        string query = b'query'
        string doc = b'doc0'
        DBNModel model = DBNModel()
        float result
        float result2
        float result3

    result = model.get_param(b'alpha', &query, &doc)[0]
    assert result > 0 and result < 1

    model.alpha_params.erase(query)
    result2 = model.get_param(b'alpha', &query, &doc)[0]
    assert result2 > 0 and result2 < 1
    assert result != result2

    result3 = model.get_param(b'alpha', &query, &doc)[0]
    assert result2 == result3


cdef test_build_e_r_vector(dbn_param *alpha_params, dbn_param *sigma_params,
                           float *gamma_param):
    cdef:
        const char *s = (
            b'[{"doc": "doc0", "click": 0, "purchase": 0},'
            b'{"doc": "doc1", "click": 1, "purchase": 0},'
            b'{"doc": "doc2", "click": 1, "purchase": 1}]'
        )
        json_object *session = json_tokener_parse(s)
        string query = b'query'
        vector[float] expected = [1, 0.4375, 0.1914, 0.0837]
        vector[float] result
        unordered_map[string, float] cr_dict
        DBNModel model = DBNModel()

    cr_dict[b'doc0'] = 0.5
    cr_dict[b'doc1'] = 0.5
    cr_dict[b'doc2'] = 0.5

    model.alpha_params = alpha_params[0]
    model.sigma_params = sigma_params[0]
    model.gamma_param = gamma_param[0]

    result = model.build_e_r_vector(session, &query, &cr_dict)
    assert_almost_equal(result, expected, decimal=4)

    json_object_put(session)


cdef test_build_X_r_vector(dbn_param *alpha_params, dbn_param *sigma_params,
                           float *gamma_param):
    cdef:
        const char *s = (
            b'[{"doc": "doc0", "click": 0, "purchase": 0},'
            b'{"doc": "doc1", "click": 1, "purchase": 0},'
            b'{"doc": "doc2", "click": 1, "purchase": 1}]'
        )
        json_object *session = json_tokener_parse(s)
        vector[float] expected = [0.73625, 0.675, 0.5, 0]
        vector[float] result
        string query = b'query'

        DBNModel model = DBNModel()

    model.alpha_params = alpha_params[0]
    model.sigma_params = sigma_params[0]
    model.gamma_param = gamma_param[0]

    result = model.build_X_r_vector(session, &query)
    assert_almost_equal(result, expected, decimal=4)

    json_object_put(session)


cdef test_build_e_r_vector_given_CP(dbn_param *alpha_params, dbn_param *sigma_params,
                                    float *gamma_param):
    cdef:
        char *s = (
            b'[{"doc": "doc0", "click": 0, "purchase": 0},'
            b'{"doc": "doc0", "click": 1, "purchase": 1},'
            b'{"doc": "doc1", "click": 0, "purchase": 0}]'
        )
        json_object *session = json_tokener_parse(s)
        vector[float] expected = [1, 0.7, 0, 0]
        vector[float] result
        string query = b'query'
        DBNModel model = DBNModel()

    model.alpha_params = alpha_params[0]
    model.sigma_params = sigma_params[0]
    model.gamma_param = gamma_param[0]

    result = model.build_e_r_vector_given_CP(session, 0, &query)
    assert_almost_equal(result, expected, decimal=4)

    result = model.build_e_r_vector_given_CP(session, 1, &query)
    expected = [1, 0, 0]
    assert_almost_equal(result, expected, decimal=4)

    result = model.build_e_r_vector_given_CP(session, 2, &query)
    expected = [1, 0.7]
    assert_almost_equal(result, expected, decimal=4)

    s = (
        b'[{"doc": "doc0", "click": 0, "purchase": 0},'
        b'{"doc": "doc0", "click": 1, "purchase": 0},'
        b'{"doc": "doc1", "click": 0, "purchase": 0}]'
    )
    session = json_tokener_parse(s)
    expected = [1, 0.7, 0.35, 0.1484]

    result = model.build_e_r_vector_given_CP(session, 0, &query)
    assert_almost_equal(result, expected, decimal=4)

    result = model.build_e_r_vector_given_CP(session, 1, &query)
    expected = [1, 0.35, 0.148484]
    assert_almost_equal(result, expected, decimal=4)

    result = model.build_e_r_vector_given_CP(session, 2, &query)
    expected = [1, 0.7]
    assert_almost_equal(result, expected, decimal=4)

    json_object_put(session)


cdef test_build_cp_p(dbn_param *alpha_params):
    cdef:
        const char *s = (
            b'[{"doc": "doc0", "click": 0, "purchase": 0},'
            b'{"doc": "doc0", "click": 1, "purchase": 1},'
            b'{"doc": "doc1", "click": 1, "purchase": 0}]'
        )
        json_object *session = json_tokener_parse(s)
        float expected = 0.005625
        float result
        string query = b'query'
        vector[float] e_r_vector_given_CP = [1, 0.6, 0.3]
        DBNModel model = DBNModel()
        unordered_map[string, float] cr_dict

    cr_dict[b'doc0'] = 0.5
    cr_dict[b'doc1'] = 0.5
    cr_dict[b'doc2'] = 0.5

    model.alpha_params = alpha_params[0]

    result = model.compute_cp_p(session, 0, &query, &e_r_vector_given_CP, &cr_dict)
    assert_almost_equal(result, expected, decimal=6)

    expected = 0.0375
    result = model.compute_cp_p(session, 1, &query, &e_r_vector_given_CP, &cr_dict)
    assert_almost_equal(result, expected, decimal=6)

    json_object_put(session)


cdef test_build_CP_vector_given_e(dbn_param *alpha_params, dbn_param *sigma_params,
                                  float *gamma_param):
    cdef:
        char *s = (
            b'[{"doc": "doc0", "click": 0, "purchase": 0},'
            b'{"doc": "doc0", "click": 1, "purchase": 1}]'
        )
        json_object *session = json_tokener_parse(s)
        DBNModel model = DBNModel()
        vector[float] result
        vector[float] expected
        unordered_map[string, float] cr_dict

    cr_dict[b'doc0'] = 0.5
    cr_dict[b'doc1'] = 0.5
    cr_dict[b'doc2'] = 0.5

    model.alpha_params = alpha_params[0]
    model.sigma_params = sigma_params[0]
    model.gamma_param = gamma_param[0]

    result = model.build_CP_vector_given_e(session, &query, &cr_dict)
    expected = [0.25]
    assert_almost_equal(result, expected, decimal=4)

    s = (
        b'[{"doc": "doc0", "click": 0, "purchase": 0},'
        b'{"doc": "doc0", "click": 1, "purchase": 0},'
        b'{"doc": "doc0", "click": 1, "purchase": 1}]'
    )
    session = json_tokener_parse(s)

    result = model.build_CP_vector_given_e(session, &query, &cr_dict)
    expected = [0.021875, 0.25]
    assert_almost_equal(result, expected, decimal=4)

    s = (
        b'[{"doc": "doc0", "click": 0, "purchase": 0},'
        b'{"doc": "doc0", "click": 1, "purchase": 0},'
        b'{"doc": "doc0", "click": 0, "purchase": 0}]'
    )
    session = json_tokener_parse(s)

    result = model.build_CP_vector_given_e(session, &query, &cr_dict)
    expected = [0.2062, 0.5]
    assert_almost_equal(result, expected, decimal=4)

    json_object_put(session)


cdef test_get_last_r():
    cdef:
        DBNModel model = DBNModel()
        char *s = (
            b'[{"doc": "doc0", "click": 0, "purchase": 0},'
            b'{"doc": "doc0", "click": 1, "purchase": 1},'
            b'{"doc": "doc1", "click": 1, "purchase": 0},'
            b'{"doc": "doc2", "click": 1, "purchase": 1}]'
        )
        json_object *session = json_tokener_parse(s)
        int result = model.get_last_r(session)
    assert result == 3

    s = (
        b'[{"doc": "doc0", "click": 0, "purchase": 0},'
        b'{"doc": "doc0", "click": 1, "purchase": 1},'
        b'{"doc": "doc1", "click": 1, "purchase": 0},'
        b'{"doc": "doc2", "click": 0, "purchase": 1}]'
    )
    session = json_tokener_parse(s)
    result = model.get_last_r(session)
    assert result == 2

    s = (
        b'[{"doc": "doc0", "click": 0, "purchase": 0},'
        b'{"doc": "doc0", "click": 0, "purchase": 1},'
        b'{"doc": "doc1", "click": 0, "purchase": 0},'
        b'{"doc": "doc2", "click": 0, "purchase": 1}]'
    )
    session = json_tokener_parse(s)
    result = model.get_last_r(session)
    assert result == 0

    json_object_put(session)


cdef test_update_tmp_alpha(dbn_param *alpha_params, dbn_param *sigma_params,
                           float *gamma_param):
    cdef:
        DBNModel model = DBNModel()
        unsigned int r = 0
        unsigned int last_r = 1
        char *s = b'{"doc": "doc0", "click": 1}'
        json_object *doc_data = json_tokener_parse(s)
        vector[float] e_r_vector = [0.5]
        vector[float] X_r_vector = [0.5]
        string query = b'query'
        unordered_map[string, vector[float]] tmp_alpha_param
        vector[float] expected

    model.alpha_params = alpha_params[0]
    model.sigma_params = sigma_params[0]
    model.gamma_param = gamma_param[0]

    tmp_alpha_param[b'doc0'] = [0, 0]

    model.update_tmp_alpha(r, &query, doc_data, &e_r_vector, &X_r_vector, last_r,
                           &tmp_alpha_param)
    assert tmp_alpha_param[b'doc0'] == [1, 1]

    r = 1
    last_r = 0
    s = b'{"doc": "doc0", "click": 0}'
    doc_data = json_tokener_parse(s)
    e_r_vector = [0.5, 0.5]
    X_r_vector = [0.5, 0.5]
    tmp_alpha_param[b'doc0'] = [0, 0]
    model.update_tmp_alpha(r, &query, doc_data, &e_r_vector, &X_r_vector, last_r,
                           &tmp_alpha_param)
    expected = [1. / 3, 1]
    assert_almost_equal(tmp_alpha_param[b'doc0'], expected, decimal=4)

    r = 1
    last_r = 2
    s = b'{"doc": "doc0", "click": 0}'
    doc_data = json_tokener_parse(s)
    e_r_vector = [0.5, 0.5]
    X_r_vector = [0.5, 0.5]
    tmp_alpha_param[b'doc0'] = [0, 0]
    model.update_tmp_alpha(r, &query, doc_data, &e_r_vector, &X_r_vector, last_r,
                           &tmp_alpha_param)
    expected = [0.0, 1]
    assert_almost_equal(tmp_alpha_param[b'doc0'], expected, decimal=4)

    json_object_put(doc_data)


cdef test_update_tmp_sigma(dbn_param *alpha_params, dbn_param *sigma_params,
                           float *gamma_param):
    cdef:
        DBNModel model = DBNModel()
        unsigned int r = 0
        unsigned int last_r = 1
        char *s = b'{"doc": "doc0", "click": 0, "purchase": 0}'
        json_object *doc_data = json_tokener_parse(s)
        vector[float] X_r_vector = [0.5, 0.5, 0.5]
        unordered_map[string, vector[float]] tmp_sigma_param
        vector[float] expected
        string query = b'query'

    model.alpha_params = alpha_params[0]
    model.sigma_params = sigma_params[0]
    model.gamma_param = gamma_param[0]

    tmp_sigma_param[b'doc0'] = [0, 0]

    model.update_tmp_sigma(&query, r, doc_data, &X_r_vector, last_r, &tmp_sigma_param)

    expected = [0, 0]
    assert_almost_equal(tmp_sigma_param[b'doc0'], expected, decimal=4)

    s = b'{"doc": "doc0", "click": 1, "purchase": 0}'
    doc_data = json_tokener_parse(s)

    model.update_tmp_sigma(&query, r, doc_data, &X_r_vector, last_r, &tmp_sigma_param)
    expected = [0, 1]
    assert_almost_equal(tmp_sigma_param[b'doc0'], expected, decimal=4)

    r = 1
    tmp_sigma_param[b'doc0'] = [0, 0]
    model.update_tmp_sigma(&query, r, doc_data, &X_r_vector, last_r, &tmp_sigma_param)
    expected = [0.6060, 1]
    assert_almost_equal(tmp_sigma_param[b'doc0'], expected, decimal=4)

    json_object_put(doc_data)


cdef test_compute_factor_last_click_lower_than_r():
    cdef:
        float result
        int r = 0
        int last_r = 0
        vector[float] cp_vector_given_e = [0.2]
        vector[float] e_r_vector_given_CP = [0.4]
        unordered_map[string, float] cr_dict
        DBNModel model = DBNModel()
        dbn_param alpha_params
        dbn_param sigma_params
        float gamma
        string query = b'query'
        bint click = False
        bint purchase = True
        string doc = b'doc0'
        Factor factor

    cr_dict[doc] = 0.1
    alpha_params[query][doc] = 0.4
    sigma_params[query][doc] = 0.4
    gamma = 0.7

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 0)
    assert_almost_equal(result, 0.6)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 0)
    assert_almost_equal(result, 0.02592)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 1)
    assert_almost_equal(result, 0.02016)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 1)
    assert_almost_equal(result, 0.012096)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 0)
    assert_almost_equal(result, 0.01728)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 0)
    assert_almost_equal(result, 0.00192)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 1)
    assert_almost_equal(result, 0.008064)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 1)
    assert_almost_equal(result, 0)


cdef test_compute_factor_last_click_higher_than_r():
    cdef:
        float result
        int r = 0
        int last_r = 1
        vector[float] cp_vector_given_e = [0.2]
        vector[float] e_r_vector_given_CP = [0.4]
        unordered_map[string, float] cr_dict
        DBNModel model = DBNModel()
        dbn_param alpha_params
        dbn_param sigma_params
        float gamma
        string query = b'query'
        bint click = False
        bint purchase = True
        string doc = b'doc0'
        Factor factor

    cr_dict[doc] = 0.1
    alpha_params[query][doc] = 0.4
    sigma_params[query][doc] = 0.4
    gamma = 0.7

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 0, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(0, 1, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 1)
    assert_almost_equal(result, 0.02016)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 1)
    assert_almost_equal(result, 0.012096)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 0, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 0)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    click = False
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 1)
    assert_almost_equal(result, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    click = True
    purchase = False
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 1)
    assert_almost_equal(result, 0.008064)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    click = True
    purchase = True
    factor = Factor()
    factor.cinit(
        r,
        last_r,
        click,
        purchase,
        alpha_params[query][doc],
        sigma_params[query][doc],
        gamma,
        cr_dict[doc],
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )
    result = factor.compute_factor(1, 1, 1)
    assert_almost_equal(result, 0)


cdef test_update_tmp_gamma():
    cdef:
        DBNModel model = DBNModel()
        int r = 0
        int last_r = 0
        char *s = b'{"doc": "doc0", "click": 1, "purchase": 0}'
        json_object *doc_data = json_tokener_parse(s)
        vector[float] cp_vector_given_e = [0.2]
        vector[float] e_r_vector_given_CP = [0.4]
        unordered_map[string, float] cr_dict
        vector[float] tmp_gamma_param
        unordered_map[string, vector[float]] tmp_alpha_param
        dbn_param alpha_params
        dbn_param sigma_params
        string query = b'query'
        float ESS_den = 0
        float ESS_0
        float ESS_1
        int i
        int j
        int k
        bint click = True
        bint purchase = False
        float alpha = 0.4
        float sigma = 0.4
        float gamma = 0.7
        float cr = 0.1

    alpha_params[query][b'doc0'] = 0.4
    sigma_params[query][b'doc0'] = 0.4
    gamma = 0.7

    model.alpha_params = alpha_params
    model.sigma_params = sigma_params
    model.gamma_param = gamma

    cr_dict[b'doc0'] = 0.1
    tmp_gamma_param = [0, 0]

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
        &e_r_vector_given_CP,
        &cp_vector_given_e
    )

    ESS_den = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ESS_den += factor.compute_factor(i, j, k)

    ESS_0 = 0.02592 / ESS_den
    ESS_1 = 0.012096 / ESS_den

    model.update_tmp_gamma(r, last_r, doc_data, &query, &cp_vector_given_e,
                           &e_r_vector_given_CP, &cr_dict, &tmp_gamma_param)

    assert_almost_equal(tmp_gamma_param[0], ESS_1)
    assert_almost_equal(tmp_gamma_param[1], ESS_1 + ESS_0)

    json_object_put(doc_data)


cdef test_update_alpha_params():
    cdef:
        DBNModel model = DBNModel()
        unordered_map[string, vector[float]] tmp_alpha_param
        string doc = b'doc0'
        string query = b'query'

    tmp_alpha_param[doc] = [1, 2]
    model.update_alpha_param(&query, &tmp_alpha_param)
    assert model.alpha_params[query][doc] == 0.5


cdef test_update_sigma_params():
    cdef:
        DBNModel model = DBNModel()
        unordered_map[string, vector[float]] tmp_sigma_param
        string doc = b'doc0'
        string query = b'query'

    tmp_sigma_param[doc] = [1, 2]
    model.update_sigma_param(&query, &tmp_sigma_param)
    assert model.sigma_params[query][doc] == 0.5


cdef test_update_gamma_param():
    cdef:
        DBNModel model = DBNModel()
        vector[float] tmp_gamma_param

    tmp_gamma_param = [1, 2]
    model.update_gamma_param(&tmp_gamma_param)
    assert model.gamma_param == 0.5

cdef test_export_judgments():

    cdef:
        DBNModel model = DBNModel()
        dbn_param alpha_params
        dbn_param sigma_params

    alpha_params[b'query0'][b'doc0'] = 0.3
    alpha_params[b'query0'][b'doc1'] = 0.4
    alpha_params[b'query0'][b'doc2'] = 0.5
    alpha_params[b'query1'][b'doc0'] = 0.6

    sigma_params[b'query0'][b'doc0'] = 0.3
    sigma_params[b'query0'][b'doc1'] = 0.4
    sigma_params[b'query0'][b'doc2'] = 0.5
    sigma_params[b'query1'][b'doc0'] = 0.6

    model.alpha_params = alpha_params
    model.sigma_params = sigma_params

    tmp_file = tempfile.NamedTemporaryFile()
    model.export_judgments(tmp_file.name)
    flag = False
    for row in open(tmp_file.name):
        result = ujson.loads(row)
        if 'query1' in result:
            assert_almost_equal(result['query1']['doc0'], 0.36)
            flag = True
        else:
            assert_almost_equal(result['query0']['doc0'], 0.09)
            assert_almost_equal(result['query0']['doc1'], 0.16)
            assert_almost_equal(result['query0']['doc2'], 0.25)

    assert flag

    tmp_file = tempfile.NamedTemporaryFile()
    filename = tmp_file.name + '.gz'
    model.export_judgments(filename)
    flag = False
    for row in gzip.GzipFile(filename, 'rb'):
        result = ujson.loads(row)
        if 'query1' in result:
            assert_almost_equal(result['query1']['doc0'], 0.36)
            flag = True
        else:
            assert_almost_equal(result['query0']['doc0'], 0.09)
            assert_almost_equal(result['query0']['doc1'], 0.16)
            assert_almost_equal(result['query0']['doc2'], 0.25)

    assert flag


cpdef run_tests():
    test_get_search_context_string()
    test_compute_cr()
    test_get_param()
    test_build_e_r_vector(&alpha_params, &sigma_params, &gamma_param)
    test_build_X_r_vector(&alpha_params, &sigma_params, &gamma_param)
    test_build_e_r_vector_given_CP(&alpha_params, &sigma_params, &gamma_param)
    test_build_cp_p(&alpha_params)
    test_build_CP_vector_given_e(&alpha_params, &sigma_params, &gamma_param)
    test_get_last_r()
    test_update_tmp_alpha(&alpha_params, &sigma_params, &gamma_param)
    test_update_tmp_sigma(&alpha_params, &sigma_params, &gamma_param)
    test_compute_factor_last_click_lower_than_r()
    test_compute_factor_last_click_higher_than_r()
    test_update_tmp_gamma()
    test_update_alpha_params()
    test_update_sigma_params()
    test_update_gamma_param()
    # test_fit()
    test_export_judgments()

if __name__ == '__main__':
    test_get_search_context_string()
    test_compute_cr()
    test_get_param()
    test_build_e_r_vector(&alpha_params, &sigma_params, &gamma_param)
    test_build_X_r_vector(&alpha_params, &sigma_params, &gamma_param)
    test_build_e_r_vector_given_CP(&alpha_params, &sigma_params, &gamma_param)
    test_build_cp_p(&alpha_params)
    test_build_CP_vector_given_e(&alpha_params, &sigma_params, &gamma_param)
    test_get_last_r()
    test_update_tmp_alpha(&alpha_params, &sigma_params, &gamma_param)
    test_update_tmp_sigma(&alpha_params, &sigma_params, &gamma_param)
    test_compute_factor_last_click_lower_than_r()
    test_compute_factor_last_click_higher_than_r()
    test_update_tmp_gamma()
    test_update_alpha_params()
    test_update_sigma_params()
    test_update_gamma_param()
    test_fit()
    test_export_judgments()
