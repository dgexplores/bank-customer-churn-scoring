import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix, roc_curve
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.utils.class_weight import compute_sample_weight

# Import feature engineer and column definitions from churn_utils
from churn_utils import (
    clean_and_normalize_columns, ChurnFeatureEngineer,
    RAW_NUMERIC_COLS, ENGINEERED_NUMERIC_COLS, CATEGORICAL_COLS,
    DROP_COLS, TARGET_COL
)

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def run_pipeline():
    print("--- Starting Churn Prediction Pipeline ---")
    
    # 1. Load data
    data_path = 'data/European_Bank.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df_raw = pd.read_csv(data_path)
    df = clean_and_normalize_columns(df_raw)
    
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Missing values check
    missing = df.isnull().sum()
    print("Missing values per column:")
    for col, count in missing.items():
        print(f"  - {col}: {count}")
    
    # Check overall churn rate
    churn_rate = df[TARGET_COL].mean()
    n_customers = len(df)
    print(f"Total customers: {n_customers}")
    print(f"Overall Churn Rate (Target = {TARGET_COL}): {churn_rate:.2%}")
    
    # 2. Train-Test Split (80/20 Stratified)
    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors='ignore')
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 3. Preprocessing Setup
    # Combine raw numeric and engineered numeric columns (since our transformer adds them)
    all_numeric_cols = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), all_numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    
    # Create directories for outputs if they don't exist
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    
    # 4. Define Models
    # Calculate scale_pos_weight for XGBoost to handle class imbalance
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / (pos_count + 1)
    
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        print("XGBoost is available and will be evaluated.")
    else:
        print("XGBoost is not installed. Skipping XGBoost.")
        
    # We will compute sample weights for GradientBoosting to address imbalance
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # 5. Evaluate and Compare Models (5-Fold Cross-Validation on Train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    best_roc_auc = -1.0
    best_model_name = ""
    best_pipeline = None
    
    print("\n--- Model Evaluation (5-Fold CV & Test Set Performance) ---")
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Build Pipeline: Feature engineering -> Preprocessor -> Estimator
        pipeline = Pipeline([
            ('feature_engineering', ChurnFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Cross validation loop
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Construct a fresh pipeline for this fold
            fold_pipeline = Pipeline([
                ('feature_engineering', ChurnFeatureEngineer()),
                ('preprocessor', preprocessor),
                ('model', clone(model))
            ])
            
            # Fit params for this fold
            fold_fit_params = {}
            if name == 'GradientBoosting':
                fold_sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
                fold_fit_params = {'model__sample_weight': fold_sample_weights}
                
            fold_pipeline.fit(X_tr, y_tr, **fold_fit_params)
            val_probs = fold_pipeline.predict_proba(X_va)[:, 1]
            cv_scores.append(roc_auc_score(y_va, val_probs))
            
        mean_cv_roc = np.mean(cv_scores)
        
        # Train on full training set
        fit_params = {}
        if name == 'GradientBoosting':
            fit_params = {'model__sample_weight': sample_weights_train}
            
        pipeline.fit(X_train, y_train, **fit_params)
        
        # Evaluate on Test Set
        y_pred = pipeline.predict(X_test)
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_roc = roc_auc_score(y_test, y_probs)
        
        print(f"  - Mean CV ROC-AUC: {mean_cv_roc:.4f}")
        print(f"  - Test ROC-AUC:    {test_roc:.4f}")
        print(f"  - Test F1-Score:   {test_f1:.4f}")
        print(f"  - Test Recall:     {test_rec:.4f}")
        
        results.append({
            'Model': name,
            'CV_ROC_AUC': mean_cv_roc,
            'Test_Accuracy': test_acc,
            'Test_Precision': test_prec,
            'Test_Recall': test_rec,
            'Test_F1': test_f1,
            'Test_ROC_AUC': test_roc
        })
        
        # Track best model by ROC-AUC
        if test_roc > best_roc_auc:
            best_roc_auc = test_roc
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model selected: {best_model_name} (Test ROC-AUC: {best_roc_auc:.4f})")
    
    # Save model comparison
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/metrics/model_comparison.csv', index=False)
    print("Model comparison saved to outputs/metrics/model_comparison.csv")
    
    # 6. Threshold Tuning on Best Model
    y_test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_probs)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    if best_idx < len(thresholds):
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.5
        
    print(f"Tuned decision threshold: {best_threshold:.4f} (Max F1: {f1_scores[best_idx]:.4f})")
    
    # Save threshold
    with open('outputs/models/threshold.json', 'w') as f:
        json.dump({'threshold': best_threshold}, f, indent=4)
        
    # Evaluate at tuned threshold
    y_pred_tuned = (y_test_probs >= best_threshold).astype(int)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    tuned_precision = precision_score(y_test, y_pred_tuned, zero_division=0)
    tuned_recall = recall_score(y_test, y_pred_tuned, zero_division=0)
    tuned_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
    tuned_roc = roc_auc_score(y_test, y_test_probs)
    
    print(f"Metrics at Tuned Threshold ({best_threshold:.4f}):")
    print(f"  - Accuracy:  {tuned_accuracy:.4f}")
    print(f"  - Precision: {tuned_precision:.4f}")
    print(f"  - Recall:    {tuned_recall:.4f}")
    print(f"  - F1-Score:  {tuned_f1:.4f}")
    
    # Save the pipeline object
    joblib.dump(best_pipeline, 'outputs/models/best_model.pkl')
    print("Best pipeline saved to outputs/models/best_model.pkl")
    
    # 7. Get Preprocessed Feature Names
    cat_transformer = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = list(cat_transformer.get_feature_names_out(CATEGORICAL_COLS))
    feature_names = all_numeric_cols + cat_feature_names
    
    # 8. Feature Importance
    print("\nExtracting feature importances...")
    model_estimator = best_pipeline.named_steps['model']
    importances = None
    
    if hasattr(model_estimator, 'feature_importances_'):
        importances = model_estimator.feature_importances_
    elif hasattr(model_estimator, 'coef_'):
        # For Logistic Regression, use absolute coefficients as importance
        importances = np.abs(model_estimator.coef_[0])
        
    if importances is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        importance_df.to_csv('outputs/metrics/feature_importance.csv', index=False)
        print("Feature importances saved to outputs/metrics/feature_importance.csv")
    else:
        print("Model does not support direct feature importances.")
        importance_df = pd.DataFrame()
        
    # 9. SHAP Summary Plot
    try:
        import shap
        print("Running SHAP TreeExplainer...")
        # Preprocess test data to feed it directly to the model inside the pipeline
        X_test_trans = best_pipeline.named_steps['preprocessor'].transform(
            best_pipeline.named_steps['feature_engineering'].transform(X_test)
        )
        X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names)
        
        # Sample for SHAP
        shap_sample = X_test_trans_df.sample(min(200, len(X_test_trans_df)), random_state=42)
        
        explainer = shap.TreeExplainer(model_estimator)
        shap_vals = explainer.shap_values(shap_sample)
        
        # Format SHAP values based on shapes
        if isinstance(shap_vals, list):
            shap_vals_class = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            shap_vals_class = shap_vals[:, :, 1]
        else:
            shap_vals_class = shap_vals
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_class, shap_sample, show=False)
        plt.title(f"SHAP Summary Plot for {best_model_name}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig('outputs/figures/shap_summary.png', bbox_inches='tight')
        plt.close()
        print("SHAP Summary plot saved.")
    except Exception as e:
        print(f"SHAP explanation failed or skipped: {e}")
        
    # 10. Generate Figures
    print("\nGenerating visual performance figures...")
    sns.set_theme(style='whitegrid', palette='muted')
    
    # 10.1 Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=df, hue=TARGET_COL, legend=False)
    plt.title("Target Column Distribution (Exited)", fontsize=12)
    plt.xlabel("Exited (0 = Retained, 1 = Churned)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('outputs/figures/target_distribution.png')
    plt.close()
    
    # 10.2 Churn by Geography and Gender
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x='Geography', y=TARGET_COL, data=df, ax=axes[0], errorbar=None)
    axes[0].set_title("Churn Rate by Geography", fontsize=12)
    axes[0].set_ylabel("Churn Rate")
    axes[0].set_ylim(0, 0.4)
    
    sns.barplot(x='Gender', y=TARGET_COL, data=df, ax=axes[1], errorbar=None)
    axes[1].set_title("Churn Rate by Gender", fontsize=12)
    axes[1].set_ylabel("Churn Rate")
    axes[1].set_ylim(0, 0.4)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/churn_by_geography_gender.png')
    plt.close()
    
    # 10.3 Age, Balance, Products vs Churn
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(x=TARGET_COL, y='Age', data=df, ax=axes[0], hue=TARGET_COL, legend=False)
    axes[0].set_title("Age Distribution by Churn", fontsize=12)
    
    sns.boxplot(x=TARGET_COL, y='Balance', data=df, ax=axes[1], hue=TARGET_COL, legend=False)
    axes[1].set_title("Balance Distribution by Churn", fontsize=12)
    
    sns.barplot(x='NumOfProducts', y=TARGET_COL, data=df, ax=axes[2], errorbar=None)
    axes[2].set_title("Churn Rate by Number of Products", fontsize=12)
    axes[2].set_ylabel("Churn Rate")
    
    plt.tight_layout()
    plt.savefig('outputs/figures/age_balance_products_vs_churn.png')
    plt.close()
    
    # 10.4 Correlation Heatmap (including engineered features)
    df_engineered = ChurnFeatureEngineer().transform(df)
    # Exclude non-numeric and drops for correlation
    corr_cols = [c for c in df_engineered.columns if c not in DROP_COLS and df_engineered[c].dtype in [np.float64, np.int64, float, int]]
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_engineered[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Heatmap (Including Engineered Features)", fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/figures/correlation_heatmap.png')
    plt.close()
    
    # 10.5 Model-comparison bar chart
    plt.figure(figsize=(10, 6))
    melted_results = results_df.melt(id_vars='Model', value_vars=['CV_ROC_AUC', 'Test_ROC_AUC', 'Test_F1'], var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=melted_results)
    plt.title("Model Comparison Across Key Metrics", fontsize=14)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('outputs/figures/model_comparison.png')
    plt.close()
    
    # 10.6 ROC curve + Confusion Matrix subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    axes[0].plot(fpr, tpr, label=f"{best_model_name} (AUC = {tuned_roc:.3f})", color='darkorange', lw=2)
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=12)
    axes[0].legend(loc="lower right")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_tuned)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False)
    axes[1].set_title(f'Confusion Matrix at Tuned Threshold ({best_threshold:.3f})', fontsize=12)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_xticklabels(['Retained (0)', 'Churned (1)'])
    axes[1].set_yticklabels(['Retained (0)', 'Churned (1)'])
    
    plt.tight_layout()
    plt.savefig('outputs/figures/roc_curve_and_confusion_matrix.png')
    plt.close()
    
    # 10.7 Feature-importance bar chart
    if not importance_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
        plt.title("Top Feature Importances (Best Model)", fontsize=14)
        plt.tight_layout()
        plt.savefig('outputs/figures/feature_importance.png')
        plt.close()
        
    # 10.8 PDP Plot for Top Features
    try:
        top_features_pdp = list(importance_df['Feature'].iloc[:3])
        fig, ax = plt.subplots(figsize=(14, 5))
        # Preprocessor returns preprocessed np arrays, so we can't easily run PartialDependenceDisplay 
        # on pipeline directly unless X contains the input pandas format that the pipeline expects.
        # This is exactly correct, the pipeline expects the raw dataframe input X!
        # So we pass X_train (pandas format) and it runs the feature_engineering and preprocessing step.
        PartialDependenceDisplay.from_estimator(best_pipeline, X_train, features=top_features_pdp, ax=ax)
        plt.suptitle("Partial Dependence Plots (Top 3 Features)", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('outputs/figures/pdp_top_features.png', bbox_inches='tight')
        plt.close()
        print("Partial Dependence Plots generated successfully.")
    except Exception as e:
        print(f"PDP generation skipped: {e}")
        
    # 11. Write outputs/metrics/insights.json
    # Churn rates by segment calculations
    geography_churn = df.groupby('Geography')[TARGET_COL].mean().to_dict()
    gender_churn = df.groupby('Gender')[TARGET_COL].mean().to_dict()
    product_churn = df.groupby('NumOfProducts')[TARGET_COL].mean().to_dict()
    # Convert keys to string for JSON serialization compatibility
    product_churn = {str(k): v for k, v in product_churn.items()}
    activity_churn = df.groupby('IsActiveMember')[TARGET_COL].mean().to_dict()
    activity_churn = {str(k): v for k, v in activity_churn.items()}
    
    top_features_list = []
    if not importance_df.empty:
        top_features_list = importance_df.head(10).to_dict(orient='records')
        
    # Best model metrics dictionary
    best_metrics = {
        'accuracy_default': float(accuracy_score(y_test, best_pipeline.predict(X_test))),
        'precision_default': float(precision_score(y_test, best_pipeline.predict(X_test), zero_division=0)),
        'recall_default': float(recall_score(y_test, best_pipeline.predict(X_test), zero_division=0)),
        'f1_default': float(f1_score(y_test, best_pipeline.predict(X_test), zero_division=0)),
        'roc_auc': float(tuned_roc),
        'accuracy_tuned': float(tuned_accuracy),
        'precision_tuned': float(tuned_precision),
        'recall_tuned': float(tuned_recall),
        'f1_tuned': float(tuned_f1)
    }
    
    insights = {
        'n_customers': int(n_customers),
        'churn_rate': float(churn_rate),
        'best_model': str(best_model_name),
        'tuned_threshold': float(best_threshold),
        'metrics': best_metrics,
        'churn_rates_by_segment': {
            'geography': geography_churn,
            'gender': gender_churn,
            'products': product_churn,
            'activity': activity_churn
        },
        'top_features': top_features_list
    }
    
    with open('outputs/metrics/insights.json', 'w') as f:
        json.dump(insights, f, indent=4)
        
    print("Insights file saved to outputs/metrics/insights.json")
    print("--- Pipeline Execution Finished Successfully ---")

if __name__ == '__main__':
    run_pipeline()
