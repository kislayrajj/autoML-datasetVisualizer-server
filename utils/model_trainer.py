from sklearn.model_selection import train_test_split
from utils.model_selector import get_model
from utils.metrics import get_metrics
from sklearn.decomposition import PCA

def train_model(df, task_type, algorithm = None):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    result = {
        "task": task_type,
        "algorithm": algorithm or "auto"
    }

    if task_type in ['classification', 'regression']:
        X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)

        model = get_model(task_type,algorithm)
        # debugging
        print("MODEL:", model)
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        model.fit(X_train,y_train)
        predictions = model.predict(X_test)

        result['metrics'] = get_metrics(task_type,y_test,predictions)
        result['predictions'] = predictions.tolist()
        result['y_test'] = y_test.tolist()

    elif task_type == 'clustering':
        model = get_model('clustering', algorithm)
        model.fit(X)
        labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)

        result['labels'] = labels.tolist()
        result['metrics'] = get_metrics('clustering', X, labels)

        # optional pca for 2d visualization
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X)
        result['pca'] = reduced.tolist()

    else:
        raise ValueError('Unsupported task type')

    return result



