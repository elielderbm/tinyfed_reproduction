from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_squared_error, mean_absolute_error

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    return acc, rec, f1

def regression_errors(y_true, y_pred_proba):
    mse = mean_squared_error(y_true, y_pred_proba)
    mae = mean_absolute_error(y_true, y_pred_proba)
    return mse, mae
