from numpy.ma.extras import average
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, log_loss, confusion_matrix, precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score

def get_metrics(task,y_true, y_pred):
    if task == 'classification':
        return {
            "accuracy": accuracy_score(y_true,y_pred),
            "precision" : precision_score(y_true,y_pred),
            "f1" : f1_score(y_true,y_pred),
            "jaccard" : jaccard_score(y_true,y_pred,average='weighted'),
            "log_loss": log_loss(y_true, y_pred, labels=list(set(y_true))),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    elif task == "regression":
        return {
            "mse": mean_squared_error(y_true,y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "r2_square": r2_score(y_true,y_pred)


        }

    elif task == 'clustering':
        return {
            "silhouette": silhouette_score(y_true,y_pred)
        }

    else:
        return {}