from sklearn.linear_model import LogisticRegression
import numpy as np

def train_edge_sign_classifier(X, y):
    """
    Train a logistic regression model for edge sign prediction
    
    Parameters:
    X: Feature matrix
    y: Labels (edge signs)
    
    Returns:
    model: Trained logistic regression model
    """
    model = LogisticRegression()
    model.fit(X, y)
    return model

def predict_edge_signs(model, X, threshold=0.5):
    """
    Predict signs for edges with a configurable threshold for positive predictions.
    
    Parameters:
    model: Trained model
    X: Feature matrix for edges
    threshold: Probability threshold for classifying edges as positive
    
    Returns:
    predictions: Predicted signs for edges
    probabilities: Predicted probabilities for positive signs
    """
    probabilities = model.predict_proba(X)[:, 1]  # Probability for the positive class
    predictions = np.where(probabilities >= threshold, 1, -1)  # Apply threshold
    return predictions, probabilities

def print_model_info(model):
    """
    Print information about the trained model, such as coefficients and intercept.
    
    Parameters:
    model: Trained model
    """
    if hasattr(model, 'coef_'):
        print("Model coefficients (weights):", model.coef_)
    if hasattr(model, 'intercept_'):
        print("Model intercept:", model.intercept_)
    else:
        print("Model does not have accessible coefficients or intercept.")