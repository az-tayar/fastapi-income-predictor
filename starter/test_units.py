from sklearn.ensemble import RandomForestClassifier

def test_train_model():
    from starter.train_model import rfc_model

    assert isinstance(rfc_model, RandomForestClassifier)


def test_compute_model_metrics():
    from starter.train_model import precision, recall, fbeta, accuracy

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert isinstance(accuracy, float)


def test_inference():
    from starter.train_model import X_test, preds

    assert len(preds) == len(X_test)

