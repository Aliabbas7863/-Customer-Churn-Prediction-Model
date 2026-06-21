import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import joblib


def load_and_clean(path):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    return df


def feature_engineering(df):
    df = df.copy()

    # Create numeric service counts (treat 'No phone service' as 'No')
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['num_services'] = df[service_cols].apply(lambda row: sum(v == 'Yes' for v in row), axis=1)

    # Support interactions proxy: number of support-like services
    df['support_count'] = df[['TechSupport', 'OnlineSecurity']].apply(lambda row: sum(v == 'Yes' for v in row), axis=1)

    # Payment patterns
    df['auto_pay'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
    df['electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)

    # Tenure buckets (simple)
    df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 1000], labels=['0-12','12-24','24-48','48+'])

    # Customer value proxy
    df['customer_value'] = df['MonthlyCharges'] * df['tenure']

    return df


def prepare_X_y(df):
    df = df.copy()
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    y = df['Churn']
    X = df.drop(columns=['Churn','customerID'] if 'customerID' in df.columns else ['Churn'])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def evaluate_model(name, model, X_test, y_test, save_dir):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probs) if probs is not None else None,
    }

    # Save classification report
    creport = classification_report(y_test, preds, digits=4)
    with open(os.path.join(save_dir, f'{name}_classification_report.txt'), 'w') as f:
        f.write(creport)

    # Save metrics
    with open(os.path.join(save_dir, f'{name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ROC curve
    if probs is not None:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC={metrics["roc_auc"]:.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{name}_roc.png'), dpi=150)
        plt.close()

    return metrics, preds, probs


def segment_customers(df, probs, out_path):
    # Assign risk by probability
    df_out = df.copy()
    df_out['churn_prob'] = probs
    def risk(p):
        if p >= 0.66:
            return 'High'
        if p >= 0.33:
            return 'Medium'
        return 'Low'
    df_out['risk_category'] = df_out['churn_prob'].apply(risk)

    # Value-based segmentation
    q = df_out['customer_value'].quantile([0.33,0.66])
    def val_seg(x):
        if x >= q.loc[0.66]:
            return 'High'
        if x >= q.loc[0.33]:
            return 'Medium'
        return 'Low'
    df_out['value_segment'] = df_out['customer_value'].apply(val_seg)

    df_out.to_csv(out_path, index=False)
    return df_out


def main():
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(csv_path):
        print('CSV not found at', csv_path)
        return

    out_dir = os.path.join('outputs', 'models')
    os.makedirs(out_dir, exist_ok=True)

    print('Loading data...')
    df = load_and_clean(csv_path)
    print('Feature engineering...')
    df_fe = feature_engineering(df)
    X, y = prepare_X_y(df_fe)
    print('Data prepared:', X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    print('Training Logistic Regression...')
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(out_dir, 'logistic_regression.joblib'))
    lr_metrics, lr_preds, lr_probs = evaluate_model('logistic', lr, X_test, y_test, out_dir)
    print('Logistic metrics:', lr_metrics)

    # Random Forest
    print('Training Random Forest...')
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(out_dir, 'random_forest.joblib'))
    rf_metrics, rf_preds, rf_probs = evaluate_model('random_forest', rf, X_test, y_test, out_dir)
    print('Random Forest metrics:', rf_metrics)

    # Choose best by roc_auc if available else f1
    def score(m):
        return (m['roc_auc'] if m.get('roc_auc') is not None else m['f1'])
    best_name = 'logistic' if score(lr_metrics) >= score(rf_metrics) else 'random_forest'
    print('Best model:', best_name)

    # Segmentation: use best model probs on full dataset
    best_model = lr if best_name == 'logistic' else rf
    # ensure X has same columns as training
    probs_full = best_model.predict_proba(X)[:,1]
    seg_out = os.path.join('outputs', 'segmentation.csv')
    seg_df = segment_customers(df_fe, probs_full, seg_out)
    print('Segmentation saved to', seg_out)

    # Save summary metrics
    summary = {'logistic': lr_metrics, 'random_forest': rf_metrics, 'best_model': best_name}
    with open(os.path.join(out_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('Pipeline complete. Outputs in outputs/models and outputs/segmentation.csv')


if __name__ == '__main__':
    main()
