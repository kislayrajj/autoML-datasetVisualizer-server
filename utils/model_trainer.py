from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# Import metrics functions needed for plot data
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import numpy as np # Import numpy for checking types if needed

# Assuming these utils are in the correct path relative to where this script is run
from utils.model_selector import get_model
from utils.metrics import get_metrics


def train_model(df, task_type, algorithm=None):
    # --- 1. Prepare Data ---
    # Assuming the last column is the target variable 'y'
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # --- 2. Initialize Result Dictionary ---
    # Store basic info
    result = {
        "task": task_type,
        "algorithm": algorithm or "default" # Provide a sensible default name if None
    }
    # Initialize plot_data dictionary - THIS IS KEY
    plot_data = {}

    # --- 3. Handle Classification and Regression ---
    if task_type in ['classification', 'regression']:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get and train model
        model = get_model(task_type, algorithm)
        # Update algorithm name if it was None initially
        if result["algorithm"] == "default":
             # Get the actual class name of the model instance
             result["algorithm"] = model.__class__.__name__

        print(f"Training model: {result['algorithm']} for task: {task_type}")
        model.fit(X_train, y_train)

        # Get predictions
        predictions = model.predict(X_test)

        # --- Calculate Metrics ---
        # get_metrics returns a dict like {"accuracy": 0.9, ...} or {"mse": 10.5, ...}
        result['metrics'] = get_metrics(task_type, y_test, predictions)

        # --- Prepare Plot Data ---
        # Convert core results to lists and add to plot_data
        plot_data['y_test'] = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
        plot_data['predictions'] = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

        # Add task-specific plot data
        if task_type == 'regression':
            # Calculate residuals
            residuals = y_test - predictions
            plot_data['residuals'] = residuals.tolist() if hasattr(residuals, 'tolist') else list(residuals)
            # Update metric names to match frontend expectation (r2_score instead of r2_square)
            if 'r2_square' in result['metrics']:
                 result['metrics']['r2_score'] = result['metrics'].pop('r2_square')


        elif task_type == 'classification':
            # Add class labels (useful for confusion matrix)
            # Ensure labels are sorted consistently
            plot_data['class_labels'] = sorted(list(map(str, set(y_test)))) # Convert to string just in case

            # Add confusion matrix to plot_data as well (optional, but frontend might look here too)
            if 'confusion_matrix' in result['metrics']:
                 plot_data['confusion_matrix'] = result['metrics']['confusion_matrix']

            # --- Calculate ROC and PR Curve data ---
            # Check if model supports predict_proba (needed for ROC/AUC/PR)
            if hasattr(model, "predict_proba"):
                try:
                    # Get probabilities for the positive class (usually class 1)
                    # Assumes binary classification for simplicity here. Adapt for multi-class if needed.
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    plot_data['roc_curve'] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": auc_score
                    }
                    # Add AUC to metrics as well, as frontend might check both places
                    result['metrics']['auc'] = auc_score


                    # Precision-Recall Curve
                    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
                    plot_data['pr_curve'] = {
                        "precision": precision_vals.tolist(),
                        "recall": recall_vals.tolist()
                    }
                except Exception as e:
                    print(f"Warning: Could not calculate ROC/PR curve data. Error: {e}")
                    # Include nulls so frontend knows data is missing
                    plot_data['roc_curve'] = None
                    plot_data['pr_curve'] = None
            else:
                print(f"Warning: Model {result['algorithm']} does not support predict_proba. Cannot generate ROC/PR curves.")
                plot_data['roc_curve'] = None
                plot_data['pr_curve'] = None


        # Add the populated plot_data dictionary to the main result
        result['plot_data'] = plot_data

    # --- 4. Handle Clustering ---
    elif task_type == 'clustering':
        model = get_model('clustering', algorithm)
         # Update algorithm name if it was None initially
        if result["algorithm"] == "default":
             result["algorithm"] = model.__class__.__name__

        print(f"Running clustering model: {result['algorithm']}")
        # Fit clustering model (usually on the whole dataset X)
        # Some models fit and predict, others just fit and have labels_
        if hasattr(model, "fit_predict"):
             labels = model.fit_predict(X)
        else:
             model.fit(X)
             labels = model.labels_

        # --- Calculate Metrics ---
        # Ensure silhouette score doesn't crash if only one cluster is found
        unique_labels = set(labels)
        if len(unique_labels) > 1:
             # Pass X and labels to get_metrics for silhouette score calculation
             result['metrics'] = get_metrics('clustering', X, labels) # Note: Swapped X and labels based on common silhouette usage
        else:
             print("Warning: Only one cluster found. Cannot calculate Silhouette Score.")
             result['metrics'] = {"silhouette_score": None} # Indicate score couldn't be calculated


        # --- Prepare Plot Data ---
        # Add labels to plot_data
        plot_data['labels'] = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

        # Perform PCA for visualization
        try:
            # Reduce dimensionality to 2 components for plotting
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(X)
            plot_data['pca'] = reduced_data.tolist() # Add PCA results to plot_data
        except Exception as e:
             print(f"Warning: PCA calculation failed. Error: {e}")
             plot_data['pca'] = None # Indicate PCA failed


        # Add the populated plot_data dictionary to the main result
        result['plot_data'] = plot_data

    # --- 5. Handle Unsupported Task ---
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # --- 6. Return Final Result ---
    # Add a final print before returning for easy debugging
    import json
    print("DEBUG: Returning result from train_model:")
    # Use default=str to handle potential non-serializable types gracefully during debug print
    # print(json.dumps(result, indent=2, default=str))
    # Be careful printing large arrays (like pca, y_test) fully in logs
    # Consider printing only shapes or snippets for large data

    return result
