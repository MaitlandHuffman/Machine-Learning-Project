# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ML libraries
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import shap

# Configure visualizations
plt.style.use('seaborn')
sns.set_palette('husl')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='Dataset.csv'):
    """Load and preprocess the TMDB dataset with enhanced feature engineering."""
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.dropna(subset=['vote_average', 'budget', 'revenue'])
    
    # Feature engineering
    df['budget_log'] = np.log1p(df['budget'])
    df['revenue_log'] = np.log1p(df['revenue'])
    df['roi'] = (df['revenue'] - df['budget']) / (df['budget'] + 1e-6)
    
    # Cap extreme values
    df['roi'] = df['roi'].clip(lower=df['roi'].quantile(0.01), 
                              upper=df['roi'].quantile(0.99))
    df['popularity'] = df['popularity'].clip(lower=df['popularity'].quantile(0.01), 
                                            upper=df['popularity'].quantile(0.99))
    
    # Additional features
    df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1e-6)
    df['revenue_per_minute'] = df['revenue'] / (df['runtime'] + 1e-6)
    df['vote_count_log'] = np.log1p(df['vote_count'])
    
    # Classification labels with more nuanced thresholds
    vote_avg_quantiles = df['vote_average'].quantile([0.33, 0.67])
    
    def classify_score(v):
        if v >= vote_avg_quantiles[0.67]:
            return 'High'
        elif v >= vote_avg_quantiles[0.33]:
            return 'Medium'
        else:
            return 'Low'
    
    df['score_class'] = df['vote_average'].apply(classify_score)
    
    return df

def prepare_features(df):
    """Prepare features with advanced preprocessing and selection."""
    # Select initial features
    features = [
        'budget_log', 'revenue_log', 'popularity', 'runtime',
        'roi', 'budget_per_minute', 'revenue_per_minute',
        'vote_count_log'
    ]
    
    X = df[features].copy()
    
    # Scale features using RobustScaler (more resistant to outliers)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_scaled

def train_optimized_classifier(X_train, y_train_class):
    """Train an optimized XGBoost classifier with hyperparameter tuning."""
    # Initial classifier with reasonable defaults
    clf = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # RandomizedSearchCV with cross-validation
    search = RandomizedSearchCV(
        clf, param_dist, n_iter=20, cv=5,
        scoring='accuracy', n_jobs=-1,
        random_state=42, verbose=1
    )
    
    # Convert one-hot encoded targets back to label encoded
    y_train_labels = y_train_class.idxmax(axis=1)
    
    # Fit the model
    search.fit(X_train, y_train_labels)
    
    print("Best parameters:", search.best_params_)
    print("Best cross-validation accuracy: {:.4f}".format(search.best_score_))
    
    return search.best_estimator_

def train_optimized_regressor(X_train, y_train_reg):
    """Train an optimized XGBoost regressor with hyperparameter tuning."""
    reg = XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    search = RandomizedSearchCV(
        reg, param_dist, n_iter=20, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=42, verbose=1
    )
    
    search.fit(X_train, y_train_reg)
    
    print("Best parameters:", search.best_params_)
    print("Best cross-validation MSE: {:.4f}".format(-search.best_score_))
    
    return search.best_estimator_

def analyze_feature_importance(model, X_test, model_type='classifier'):
    """Analyze and visualize feature importance using SHAP values."""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 6))
    
    # For classifier, we'll look at the 'High' class
    if model_type == 'classifier':
        shap_values_class = shap_values[2]  # For 'High' class
        shap.summary_plot(shap_values_class, X_test,
                         plot_type='bar',
                         title='Feature Importance (Classification)')
    else:
        shap.summary_plot(shap_values, X_test,
                         plot_type='bar',
                         title='Feature Importance (Regression)')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features and targets
    X = prepare_features(df)
    y_class = pd.get_dummies(df['score_class'])  # One-hot encoding for classification
    y_reg = df['vote_average']  # Regression target
    
    # Split data with stratification for classification
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, stratify=y_class['High'], random_state=42
    )
    
    # Split data for regression
    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    # Train and evaluate classifier
    print("\nTraining classifier...")
    best_clf = train_optimized_classifier(X_train, y_train_class)
    
    # Evaluate classifier
    y_test_labels = y_test_class.idxmax(axis=1)
    y_pred_class = best_clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_class))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test_labels, y_pred_class),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.title('Confusion Matrix')
    plt.show()
    
    # Train and evaluate regressor
    print("\nTraining regressor...")
    best_reg = train_optimized_regressor(X_train, y_train_reg)
    
    # Evaluate regressor
    y_pred_reg = best_reg.predict(X_test)
    
    print("\nRegression Metrics:")
    print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.4f}")
    print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    plt.plot([y_test_reg.min(), y_test_reg.max()],
             [y_test_reg.min(), y_test_reg.max()],
             'r--', lw=2)
    plt.xlabel('Actual Vote Average')
    plt.ylabel('Predicted Vote Average')
    plt.title('Actual vs Predicted Vote Average')
    plt.tight_layout()
    plt.show()
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    print("Classification Model:")
    analyze_feature_importance(best_clf, X_test, 'classifier')
    print("\nRegression Model:")
    analyze_feature_importance(best_reg, X_test, 'regression')

if __name__ == "__main__":
    main() 