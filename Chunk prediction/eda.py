import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    # clean TotalCharges which may be blank and treated as object
    if df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    return df


def overview(df):
    print('Shape:', df.shape)
    print('\nDtypes:\n', df.dtypes)
    print('\nMissing values:\n', df.isna().sum())
    print('\nSample rows:\n', df.head().to_string(index=False))


def save_plot(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eda_plots(df, out_dir):
    ensure_dir(out_dir)

    # Churn distribution
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, palette='pastel', ax=ax)
    ax.set_title('Churn Distribution')
    save_plot(fig, out_dir, 'churn_distribution')

    # Numeric histograms
    nums = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in nums:
        if col in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            save_plot(fig, out_dir, f'dist_{col}')

    # Correlation heatmap for numeric features
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Numeric Correlation')
        save_plot(fig, out_dir, 'correlation_heatmap')

    # Categorical features vs churn: Contract, InternetService, PaymentMethod
    cats = ['Contract', 'InternetService', 'PaymentMethod']
    for col in cats:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.countplot(x=col, hue='Churn', data=df, palette='Set2', ax=ax)
            ax.set_title(f'{col} by Churn')
            plt.xticks(rotation=30)
            save_plot(fig, out_dir, f'{col}_by_churn')


def main():
    csv_path = os.path.join('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    out_dir = os.path.join('outputs', 'eda_plots')
    if not os.path.exists(csv_path):
        print(f'CSV not found at {csv_path}. Please run from repository root.')
        return

    df = load_data(csv_path)
    overview(df)
    eda_plots(df, out_dir)
    print(f'EDA complete. Plots saved to {out_dir}')


if __name__ == '__main__':
    main()
