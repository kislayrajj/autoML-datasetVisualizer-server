from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import  MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN

def get_model(task, algorithm = None):
    if task == "classification":
        return {
            'logistic_regression' : LogisticRegression(),
            "decision_tree" : DecisionTreeClassifier(),
            "svm" : SVC(probability=True),
            "knn" : KNeighborsClassifier(),
            "mlp" : MLPClassifier(),
            "random_forest":RandomForestClassifier()
        }.get(algorithm,RandomForestClassifier())

    elif task == 'regression':
        return {
            "linear_regression":LinearRegression(),
            "decision_tree":DecisionTreeRegressor(),
            "svm" : SVR(),
            "knn" : KNeighborsRegressor(),
            "mlp" : MLPRegressor(max_iter=500),
            "random_forest":RandomForestRegressor()
        }.get(algorithm, RandomForestRegressor())

    elif task=="clustering":
        return {
            'kmeans': KMeans(n_clusters=3),
            "hierarchical" : AgglomerativeClustering(n_clusters=3),
            "dbscan" : DBSCAN()
        }.get(algorithm, KMeans(n_clusters=3))

    else:
        raise ValueError("Unknown task type")