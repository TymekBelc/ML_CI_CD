from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    preds, y_test = train_and_predict()
    assert len(preds) > 0 and len(preds) == len(y_test)

def test_predictions_value_range():
    preds, _ = train_and_predict()
    for val in preds:
        assert val in [0, 1, 2], f"Unexpected class: {val}"

def test_model_accuracy():
    preds, y_test = train_and_predict()
    acc = get_accuracy(preds, y_test)
    assert acc >= 0.7, f"Accuracy too low: {acc}"
