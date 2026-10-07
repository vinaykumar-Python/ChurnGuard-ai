import churnguard_app as churn_app


def test_health():
    response = churn_app.app.test_client().get('/api/health')
    assert response.status_code == 200
    assert 'model_loaded' in response.json


def test_login_page():
    response = churn_app.app.test_client().get('/login')
    assert response.status_code == 200
    assert 'Sign in' in response.text


def test_protected_route_redirects():
    response = churn_app.app.test_client().get('/predict', follow_redirects=False)
    assert response.status_code == 302
    assert '/login' in response.headers['Location']


def test_api_docs_removed_from_business_ui():
    response = churn_app.app.test_client().get('/api/docs')
    assert response.status_code == 404


def test_prediction_probability_range():
    assert churn_app.MODEL is not None
    sample = churn_app.DF.iloc[0].drop(labels=['CustomerID', 'Churn']).to_dict()
    values, errors = churn_app.parse_customer(sample)
    assert not errors
    from app.ml import predict_customer
    probability = predict_customer(churn_app.MODEL, values)
    assert 0 <= probability <= 1
