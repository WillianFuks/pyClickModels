from libcpp.string cimport string

from pyClickModels.DBN2 cimport DBNModel
from pyClickModels.jsonc cimport (json_object, json_tokener_parse,
                                 json_object_object_get_ex, json_object_get_string,
                                 json_object_get_object, lh_table, lh_entry,
                                 json_object_array_length, json_object_array_get_idx)
# from .conftest import build_DBN_test_data
# from numpy.testing import assert_almost_equal, assert_allclose


# sessions = [
    # {
        # 'sessionID': [
            # {"doc": "doc0", "click": 0, "purchase": 0},
            # {"doc": "doc1", "click": 1, "purchase": 0},
            # {"doc": "doc2", "click": 1, "purchase": 1}
        # ]
    # },
    # {
        # 'sessionID': [
            # {"doc": "doc0", "click": 0, "purchase": 0},
            # {"doc": "doc1", "click": 1, "purchase": 0},
        # ]
    # },

# ]

# cr_dict = {'doc0': 0.5, 'doc1': 0.5, 'doc2': 0.5}

# tmp_vars = {
    # 'alpha': 0.5,
    # 'sigma': 0.5,
    # 'gamma': 0.5
# }
# dbn_params = {
    # 'query': {
        # 'doc0': tmp_vars,
        # 'doc1': tmp_vars,
        # 'doc2': tmp_vars
    # }
# }
# dbn_params['gamma'] = 0.7


# def test_fit():
    # model = DBNModel()
    # gamma, params, tmp_folder = build_DBN_test_data(users=1000, docs=10, queries=2)
    # model.fit(tmp_folder.name, processes=0, iters=1)
    # print('model dbn[gamma]: ', model.dbn_params['gamma'])
    # print('real gamma: ', gamma)
    # print('dbn keys: ', model.dbn_params.keys())

    # print('dbn_params[alpha] ', model.dbn_params['0_L_north']['0']['alpha'])
    # print('params alpha ', params[0][0][0])
    # print('dbn_params[sigma]', model.dbn_params['0_L_north']['0']['sigma'])
    # print('params sigma ', params[0][0][1])

    # assert_allclose(model.dbn_params['gamma'], gamma, atol=.1)
    # assert_allclose(model.dbn_params['0_L_north']['0']['alpha'], params[0][0][0],
                               # atol=.15)
    # assert_allclose(model.dbn_params['0_L_north']['0']['sigma'], params[0][0][1],
                               # atol=.15)


cdef bint test_get_search_context_string():
    cdef DBNModel model = DBNModel()
    cdef json_object *search_keys = json_tokener_parse(b"{'search_term': 'query'}")
    assert False
    # cdef lh_table *tbl = json_object_get_object(search_keys)
    # cdef string r = model.get_search_context_string(tbl)
    # cdef string expected = b'query'
    # assert r == expected

    # search_keys = {'search_term': 'query', 'key0': 'value0', 'key1': 'value1'}
    # r = model.get_search_context_string(search_keys)
    # assert r == 'query_value0_value1'


# def test_compute_cr():
    # model = DBNModel()
    # query = 'query'
    # cr_dict = {}
    # model.compute_cr(query, sessions, cr_dict)
    # expected = {'query': {'doc0': 0.0, 'doc1': 0.0, 'doc2': 1.}}
    # assert expected == cr_dict


# def test_build_e_r_array():
    # session = list(sessions[0].values())[0]
    # model = DBNModel()
    # r = model.build_e_r_array(session, 'query', dbn_params, cr_dict)
    # expected = [1, 0.4375, 0.1914, 0.0837]
    # assert_almost_equal(list(r), expected, decimal=4)


# def test_build_X_r_array():
    # session = list(sessions[0].values())[0]
    # model = DBNModel()
    # r = model.build_X_r_array(session, 'query', dbn_params)
    # expected = [0.73625, 0.675, 0.5, 0]
    # assert_almost_equal(list(r), expected, decimal=4)


# def test_build_e_r_array_given_CP():
    # s = [
        # {"doc": "doc0", "click": 0, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 1},
        # {"doc": "doc1", "click": 0, "purchase": 0}
    # ]
    # expected = [1, 0.7, 0, 0]
    # model = DBNModel()
    # r = model.build_e_r_array_given_CP(s, 'query', dbn_params)
    # assert_almost_equal(list(r), expected, decimal=4)

    # s = [
        # {"doc": "doc0", "click": 0, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 0},
        # {"doc": "doc1", "click": 0, "purchase": 0}
    # ]
    # expected = [1, 0.7, 0.35, 0.1484]
    # r = model.build_e_r_array_given_CP(s, 'query', dbn_params)
    # assert_almost_equal(list(r), expected, decimal=4)


# def test_build_cp_p():
    # s = [
        # {"doc": "doc0", "click": 0, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 1},
        # {"doc": "doc1", "click": 1, "purchase": 0}
    # ]
    # e_r_array_given_CP = array.array('f', [1, 0.6, 0.3])
    # model = DBNModel()
    # r = model.compute_cp_p(s, 'query', dbn_params, e_r_array_given_CP, cr_dict)
    # expected = 0.005625
    # assert_almost_equal(r, expected, decimal=6)


# def test_build_CP_array_given_e():
    # s = [
        # {"doc": "doc0", "click": 0, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 1}
    # ]
    # model = DBNModel()
    # r = model.build_CP_array_given_e(s, 'query', dbn_params, cr_dict)
    # expected = [0.25]
    # assert_almost_equal(list(r), expected, decimal=4)

    # s = [
        # {"doc": "doc0", "click": 0, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 1}
    # ]
    # r = model.build_CP_array_given_e(s, 'query', dbn_params, cr_dict)
    # expected = [0.021875, 0.25]
    # assert_almost_equal(list(r), expected, decimal=4)

    # s = [
        # {"doc": "doc0", "click": 0, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 0},
        # {"doc": "doc0", "click": 0, "purchase": 0}
    # ]
    # expected = [0.2062, 0.5]
    # r = model.build_CP_array_given_e(s, 'query', dbn_params, cr_dict)
    # assert_almost_equal(list(r), expected, decimal=4)


# def test_get_last_r():
    # model = DBNModel()
    # s = [
        # {"doc": "doc0", "click": 0, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 1},
        # {"doc": "doc1", "click": 1, "purchase": 0},
        # {"doc": "doc2", "click": 1, "purchase": 1}
    # ]
    # r = model.get_last_r(s)
    # assert r == 3

    # s = [
        # {"doc": "doc0", "click": 0, "purchase": 0},
        # {"doc": "doc0", "click": 1, "purchase": 1},
        # {"doc": "doc1", "click": 1, "purchase": 0},
        # {"doc": "doc2", "click": 0, "purchase": 1}
    # ]
    # r = model.get_last_r(s)
    # assert r == 2


# def test_update_alpha():
    # model = DBNModel()
    # r = 0
    # last_r = 1
    # doc_data = {'doc': 'doc0', 'click': 1}
    # e_r_array = array.array('f', [0.5])
    # X_r_array = array.array('f', [0.5])
    # tmp_vars = {
        # 'doc0': {'alpha': [0, 0]}
    # }
    # model.update_alpha(r, 'query', doc_data, e_r_array, X_r_array, last_r, tmp_vars,
                       # dbn_params)
    # assert tmp_vars['doc0']['alpha'] == [1, 1]

    # r = 1
    # last_r = 0
    # doc_data = {'doc': 'doc0', 'click': 0}
    # e_r_array = array.array('f', [0.5, 0.5])
    # X_r_array = array.array('f', [0.5, 0.5])
    # tmp_vars = {
        # 'doc0': {'alpha': [0, 0]}
    # }
    # model.update_alpha(r, 'query', doc_data, e_r_array, X_r_array, last_r, tmp_vars,
                       # dbn_params)
    # expected = [1. / 3, 1]
    # assert_almost_equal(tmp_vars['doc0']['alpha'], expected, decimal=4)

    # r = 1
    # last_r = 2
    # doc_data = {'doc': 'doc0', 'click': 0}
    # e_r_array = array.array('f', [0.5, 0.5])
    # X_r_array = array.array('f', [0.5, 0.5])
    # tmp_vars = {
        # 'doc0': {'alpha': [0, 0]}
    # }
    # model.update_alpha(r, 'query', doc_data, e_r_array, X_r_array, last_r, tmp_vars,
                       # dbn_params)
    # expected = [0.0, 1]
    # assert_almost_equal(tmp_vars['doc0']['alpha'], expected, decimal=4)


# def test_update_sigma():
    # model = DBNModel()
    # r = 0
    # last_r = 1
    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # X_r_array = array.array('f', [0.5, 0.5, 0.5])
    # tmp_vars = {
        # 'doc0': {'sigma': [0, 0]}
    # }
    # model.update_sigma('query', r, doc_data, X_r_array, last_r, tmp_vars, dbn_params)
    # expected = [0, 0]
    # assert_almost_equal(tmp_vars['doc0']['sigma'], expected, decimal=4)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # tmp_vars = {
        # 'doc0': {'sigma': [0, 0]}
    # }
    # model.update_sigma('query', r, doc_data, X_r_array, last_r, tmp_vars, dbn_params)
    # expected = [0, 1]
    # assert_almost_equal(tmp_vars['doc0']['sigma'], expected, decimal=4)

    # r = 1
    # tmp_vars = {
        # 'doc0': {'sigma': [0, 0]}
    # }
    # model.update_sigma('query', r, doc_data, X_r_array, last_r, tmp_vars, dbn_params)
    # expected = [0.6060, 1]
    # assert_almost_equal(tmp_vars['doc0']['sigma'], expected, decimal=4)


# def test_compute_factor_last_click_lower_than_r():
    # r = 0
    # cp_array_given_e = array.array('f', [0.2])
    # e_r_array_given_CP = array.array('f', [0.4])
    # cr_dict = {'doc0': 0.1}
    # last_r = 0
    # model = DBNModel()

    # tmp_vars = {
        # 'alpha': 0.4,
        # 'sigma': 0.4,
    # }
    # dbn_params = {
        # 'query': {
            # 'doc0': tmp_vars
        # }
    # }
    # dbn_params['gamma'] = 0.7

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 0)
    # assert_almost_equal(r, 0.6)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 0)
    # assert_almost_equal(r, 0.02592)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 1)
    # assert_almost_equal(r, 0.02016)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 1)
    # assert_almost_equal(r, 0.012096)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 0)
    # assert_almost_equal(r, 0.01728)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 0)
    # assert_almost_equal(r, 0.00192)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 1)
    # assert_almost_equal(r, 0.008064)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 1)
    # assert_almost_equal(r, 0)


# def test_compute_factor_last_click_higher_than_r():
    # r = 0
    # cp_array_given_e = array.array('f', [0.2])
    # e_r_array_given_CP = array.array('f', [0.4])
    # cr_dict = {'doc0': 0.1}
    # last_r = 1
    # model = DBNModel()

    # dbn_params = {
        # 'query': {
            # 'doc0': {'alpha': 0.4, 'sigma': 0.4}
        # }
    # }
    # dbn_params['gamma'] = 0.7

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 0, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(0, 1, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 1)
    # assert_almost_equal(r, 0.02016)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 1)
    # assert_almost_equal(r, 0.012096)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 0, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 0)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 0, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 1)
    # assert_almost_equal(r, 0)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 1)
    # assert_almost_equal(r, 0.008064)

    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 1}
    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )
    # r = factor.compute_factor(1, 1, 1)
    # assert_almost_equal(r, 0)


# def test_update_gamma():
    # model = DBNModel()
    # r = 0
    # doc_data = {'doc': 'doc0', 'click': 1, 'purchase': 0}
    # cp_array_given_e = array.array('f', [0.2])
    # e_r_array_given_CP = array.array('f', [0.4])
    # cr_dict = {'doc0': 0.1}
    # last_r = 0
    # tmp_vars = {
        # 'doc0': {'alpha': [0, 0], 'sigma': [0, 0]},
        # 'gamma': [0, 0]
    # }
    # dbn_params = {
        # 'query': {
            # 'doc0': {'alpha': 0.4, 'sigma': 0.4}
        # }
    # }
    # dbn_params['gamma'] = 0.7

    # factor = Factor(
        # r,
        # last_r,
        # doc_data['doc'],
        # 'query',
        # doc_data['click'],
        # doc_data['purchase'],
        # dbn_params,
        # cr_dict[doc_data['doc']],
        # e_r_array_given_CP,
        # cp_array_given_e
    # )

    # ESS_den = 0
    # for i in range(2):
        # for j in range(2):
            # for k in range(2):
                # ESS_den += factor.compute_factor(i, j, k)

    # ESS_0 = 0.02592 / ESS_den
    # ESS_1 = 0.012096 / ESS_den

    # model.update_gamma(r, last_r, doc_data, 'query', dbn_params, cp_array_given_e,
                        # e_r_array_given_CP, cr_dict, tmp_vars)

    # assert_almost_equal(tmp_vars['gamma'][0], ESS_1)
    # assert_almost_equal(tmp_vars['gamma'][1], ESS_1 + ESS_0)


# def test_update_dbn_params():
    # tmp_vars = {
        # 'doc0': {
            # 'alpha': [1, 2],
            # 'sigma': [1, 3]
        # },
        # 'gamma': [1, 10]
    # }
    # dbn_params = {
        # 'query': {
            # 'doc0': {}
        # }
    # }
    # model = DBNModel()
    # model.update_dbn_params('query', dbn_params, tmp_vars)
    # assert dbn_params['gamma'] == 0.1
    # assert dbn_params['query']['doc0']['alpha'] == 0.5
    # assert dbn_params['query']['doc0']['sigma'] == 1. / 3


test_get_search_context_string()
# test_compute_cr()
# test_build_e_r_array()
# test_build_X_r_array()
# test_build_e_r_array_given_CP()
# test_build_cp_p()
# test_build_CP_array_given_e()
# test_get_last_r()
# test_update_alpha()
# test_update_sigma()
# test_compute_factor_last_click_lower_than_r()
# test_compute_factor_last_click_higher_than_r()
# test_update_gamma()
# test_update_dbn_params()
# test_fit()
