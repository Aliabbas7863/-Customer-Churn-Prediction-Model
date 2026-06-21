import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os


def load_and_clean(path):
    df = pd.read_csv(path)
    # Convert TotalCharges to numeric (some rows may be blank)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Drop rows with missing TotalCharges
    df = df.dropna(subset=['TotalCharges'])
    # Drop customerID (not useful for prediction)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df


def preprocess(df):
    # Convert target to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # One-hot encode categorical features (simple and readable)
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train_and_evaluate(X, y, save_dir=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=4)
    cm = confusion_matrix(y_test, preds)

    print(f'Accuracy: {acc:.4f}')
    print('\nClassification report:')
    print(report)
    print('\nConfusion matrix:')
    print(cm)

    results = {
        'accuracy': float(acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # save classification report
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(f'Accuracy: {acc:.4f}\n\n')
            f.write(report)

        # save confusion matrix plot
        fig, ax = plt.subplots()
        cax = ax.matshow(np.array(cm), cmap='Blues')
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center')
        fig.colorbar(cax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig)

    return model, results


def main():
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(csv_path):
        print(f'CSV not found at {csv_path}. Run from project root.')
        return

    print('Loading and cleaning data...')
    df = load_and_clean(csv_path)
    print(f'Data shape after cleaning: {df.shape}')

    print('Preprocessing...')
    X, y = preprocess(df)
    print(f'Feature matrix shape: {X.shape}')

    print('Training and evaluating model...')
    model = train_and_evaluate(X, y)

    print('Done. If you want, I can save the model or try other algorithms.')


if __name__ == '__main__':
    main()
