from pyClickModels.DBN import DBNModel


def test_DBN_update_params(build_DBN_test_data):
    persistence, params, tmp_folder = build_DBN_test_data(users=10, docs=10)
    dbn = DBNModel()
    _ = dbn.fit(tmp_folder.name)
    assert False
