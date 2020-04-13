import pytest
import gzip
import json
import tempfile
import numpy as np


@pytest.mark.slow
def build_DBN_test_data(users=10, docs=10, queries=2):
    # first z column is alpha, second is sigma and third is purchase rate.
    params = np.random.random(size=(queries, docs, 3))
    persistence = 0.7

    final_result = []
    for q in range(queries):
        inner_result = {
            "search_keys": {
                "search_term": q,
                "region": "north",
                "favorite_size": "L"
            },
            "judgment_keys": []
        }
        for u in range(users):
            session = []
            counter = 0
            tmp_docs = list(range(docs))
            np.random.shuffle(tmp_docs)
            stopped_examining = False
            while True:
                counter += 1
                if counter > docs:
                    break
                doc = tmp_docs.pop()
                if stopped_examining:
                    data = {
                        'click': 0,
                        'purchase': 0,
                        'doc': str(doc)
                    }
                    session.append(data)
                    continue
                persist = np.random.random()
                satisfied = np.random.random()
                click_event = np.random.random()
                purchase_event = np.random.random()
                observed_click = 1 if click_event < params[q, doc, 0] else 0
                observed_purchase = (
                    1 if observed_click and purchase_event < params[q, doc, 2] else 0
                )
                data = {
                    'click': observed_click,
                    'purchase': observed_purchase,
                    'doc': str(doc)
                }
                session.append(data)
                # if clicked then there's chance user is satisfied
                if observed_click:
                    # user is certainly satisfied
                    if observed_purchase:
                        stopped_examining = True
                    if satisfied < params[q, doc, 1]:
                        stopped_examining = True
                    else:
                        if persist > persistence:
                            stopped_examining = True
                # if didn't click then only continue browsing given persistence
                else:
                    if persist > persistence:
                        stopped_examining = True
            inner_result['judgment_keys'].append({'session': session})
        final_result.append(inner_result)
    tmp_folder = tempfile.TemporaryDirectory()
    tmp_folder.name = '/tmp'
    half_results = int(len(final_result) / 2)
    with gzip.GzipFile(tmp_folder.name + '/judgments_model_test_data_1.gz', 'wb') as f:
        for row in final_result[:half_results]:
            f.write(json.dumps(row).encode() + '\n'.encode())

    with gzip.GzipFile(tmp_folder.name + '/judgments_model_test_data_2.gz', 'wb') as f:
        for row in final_result[half_results:]:
            f.write(json.dumps(row).encode() + '\n'.encode())
    return persistence, params, tmp_folder


@pytest.fixture
def sessions():
    sessions = [
        {
            'sessionID': [
                {"doc": "doc0", "click": 0, "purchase": 0},
                {"doc": "doc1", "click": 1, "purchase": 0},
                {"doc": "doc2", "click": 1, "purchase": 1}
            ]
        },
        {
            'sessionID': [
                {"doc": "doc0", "click": 0, "purchase": 0},
                {"doc": "doc1", "click": 1, "purchase": 0}
            ]
        },

    ]
    return sessions
