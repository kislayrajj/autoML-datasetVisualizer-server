from flask import Blueprint, jsonify

models_bp = Blueprint('models', __name__)

@models_bp.route("/models")
def get_available_models():
    return jsonify({
        "classification": [
            "logistic_regression",
            "decision_tree",
            "svm",
            "knn",
            "mlp",
            "random_forest"
        ],
        "regression": [
            "linear_regression",
            "decision_tree",
            "svm",
            "knn",
            "mlp",
            "random_forest"
        ],
        "clustering": [
            "kmeans",
            "hierarchical",
            "dbscan"
        ]
    })
