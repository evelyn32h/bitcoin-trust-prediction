from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

def train_edge_sign_classifier(X, y, model_type='logistic', **kwargs):
    """
    Train a classifier for edge sign prediction.
    Enhanced with better convergence handling for weighted features.
    
    Parameters:
    X: Feature matrix
    y: Labels (edge signs)
    model_type: Type of classifier ('logistic', 'svm', 'rf')
    **kwargs: Additional model parameters
    
    Returns:
    model: Trained classifier
    """
    print(f"Training {model_type} classifier on {X.shape[0]} samples with {X.shape[1]} features")
    
    if model_type == 'logistic':
        # Enhanced parameters for better convergence with weighted features
        default_params = {
            'max_iter': 2000,  # Increased for weighted features
            'solver': 'lbfgs',  # Good for small-medium datasets
            'random_state': 42,
            'class_weight': 'balanced'  # Handle class imbalance
        }
        default_params.update(kwargs)
        model = LogisticRegression(**default_params)
        
    elif model_type == 'svm':
        from sklearn.svm import SVC
        default_params = {
            'random_state': 42,
            'probability': True,
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        model = SVC(**default_params)
        
    elif model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        default_params = {
            'random_state': 42,
            'n_estimators': 100,
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        model = RandomForestClassifier(**default_params)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    try:
        model.fit(X, y)
        
        # Report training performance
        from sklearn.metrics import accuracy_score
        y_pred_train = model.predict(X)
        train_accuracy = accuracy_score(y, y_pred_train)
        print(f"Training accuracy: {train_accuracy:.3f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    return model

def predict_edge_signs(model, X, threshold=0.5):
    """
    Predict signs for edges with configurable threshold.
    
    Parameters:
    model: Trained model
    X: Feature matrix for edges
    threshold: Probability threshold for positive class predictions
    
    Returns:
    predictions: Predicted signs for edges (-1 or 1)
    probabilities: Predicted probabilities for positive class
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]  # Probability for positive class
        predictions = np.where(probabilities >= threshold, 1, -1)
    else:
        # Model doesn't support probabilities
        predictions = model.predict(X)
        # Convert to -1/1 format if needed
        predictions = np.where(predictions <= 0, -1, 1)
        # Create dummy probabilities
        probabilities = np.where(predictions == 1, 0.7, 0.3)
    
    return predictions, probabilities

def scale_training_features(X_train, scaler_type='standard'):
    """
    Fit a scaler on training features and return scaled features and fitted scaler.
    
    Parameters:
        X_train (array-like): Training feature matrix.
        scaler_type (str): Type of scaler ('standard', 'robust', 'maxabs')
    Returns:
        tuple: (X_train_scaled, scaler)
    """
    print(f"Scaling training features using {scaler_type} scaler")
    
    if scaler_type == 'robust':
        scaler = RobustScaler()  # Good for outliers
    elif scaler_type == 'maxabs':
        scaler = MaxAbsScaler()  # Good for sparse data
    else:
        scaler = StandardScaler()  # Default
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"Feature scaling completed:")
    print(f"  Original range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  Scaled range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    
    return X_train_scaled, scaler

def scale_testing_features(X_test, scaler):
    """
    Scale test features using a pre-fitted scaler.
    
    Parameters:
        X_test (array-like): Test feature matrix.
        scaler: Pre-fitted scaler from training.
    Returns:
        array: Scaled test feature matrix.
    """
    return scaler.transform(X_test)

def print_model_info(model):
    """
    Print information about the trained model.
    
    Parameters:
    model: Trained model
    """
    print("Model Information:")
    if hasattr(model, 'coef_'):
        print(f"  Coefficients shape: {model.coef_.shape}")
        print(f"  Coefficient range: [{model.coef_.min():.3f}, {model.coef_.max():.3f}]")
    if hasattr(model, 'intercept_'):
        print(f"  Intercept: {model.intercept_}")
    if hasattr(model, 'n_iter_'):
        print(f"  Iterations: {model.n_iter_}")
    else:
        print("  Model information not available")

def find_optimal_threshold(model, X_val, y_val, scaler=None, metric='f1'):
    """
    Find optimal threshold for binary classification using validation data.
    
    Parameters:
    model: Trained model with predict_proba method
    X_val: Validation features
    y_val: Validation labels
    scaler: Optional fitted scaler
    metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
    
    Returns:
    float: Optimal threshold
    """
    if not hasattr(model, 'predict_proba'):
        print("Model doesn't support probability prediction, using default threshold 0.5")
        return 0.5
    
    # Scale features if scaler provided
    if scaler is not None:
        X_val_scaled = scale_testing_features(X_val, scaler)
    else:
        X_val_scaled = X_val
    
    # Get prediction probabilities
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    
    # Try different thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = 0
    best_threshold = 0.5
    
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        # Convert -1/1 labels to 0/1 for sklearn metrics
        y_val_binary = (y_val == 1).astype(int)
        
        try:
            if metric == 'f1':
                score = f1_score(y_val_binary, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_val_binary, y_pred)
            elif metric == 'precision':
                score = precision_score(y_val_binary, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val_binary, y_pred, zero_division=0)
            else:
                score = f1_score(y_val_binary, y_pred, zero_division=0)
        except:
            score = 0
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f} (optimized for {metric})")
    print(f"Best {metric} score: {best_score:.3f}")
    
    return best_threshold