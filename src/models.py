from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

def scale_training_features(X_train):
    """
    Fit a StandardScaler on training features and return the scaler and scaled features.
    
    Parameters:
        X_train (array-like): Training feature matrix.
    Returns:
        tuple: (X_train_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, scaler

def scale_test_features(X_test, scaler):
    """
    Scale test features using a pre-fitted scaler.
    
    Parameters:
        X_test (array-like): Test feature matrix.
        scaler (StandardScaler): Pre-fitted scaler.
    Returns:
        array: Scaled test feature matrix.
    """
    return scaler.transform(X_test)