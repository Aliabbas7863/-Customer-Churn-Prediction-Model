# Telco Customer Churn Prediction

This project is a complete machine learning workflow for predicting customer churn in the Telco dataset.

It explains how the data is cleaned, analyzed, modeled, and visualized. The project also shows how customers can be grouped by churn risk and customer value.

## What this project does

1. Cleans the dataset and prepares it for machine learning.
2. Performs exploratory data analysis to understand churn patterns.
3. Trains a simple baseline model using logistic regression.
4. Runs a full pipeline with logistic regression and random forest.
5. Evaluates model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
6. Creates customer segments based on churn probability and customer value.
7. Shows the results in a Streamlit dashboard.

## Project files

- `eda.py` — generates EDA plots and saves them in `outputs/eda_plots`
- `simple_model.py` — trains a simple logistic regression model
- `model_pipeline.py` — runs the full modeling pipeline and segmentation
- `streamlit_app.py` — launches the interactive dashboard
- `compile_report.py` — compiles analysis and model outputs
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset used in the project

## Technologies used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- SHAP
- Joblib

## Output folders

After running the scripts, the main outputs are saved here:

- `outputs/models` — trained models, metrics, and classification reports
- `outputs/segmentation.csv` — customer risk and value segmentation
- `outputs/eda_plots` — EDA charts and plots

## Screenshots

Add your project screenshots here to make the repository more attractive on GitHub.

Example screenshots already in this repository:

- [Screenshot 1](Screenshot_2026-06-14_17-11-21.png)
- [Screenshot 2](Screenshot_2026-06-14_17-12-17.png)
- [Screenshot 3](Screenshot_2026-06-14_17-12-42.png)
- [Screenshot 4](Screenshot_2026-06-14_17-14-42.png)
- [Screenshot 5](Screenshot_2026-06-14_17-15-17.png)

## How to set up the project

Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## How to run the project

### 1) Run exploratory data analysis

```bash
python eda.py
```

This saves plots into the `outputs/eda_plots` folder.

### 2) Run the simple churn model

```bash
python simple_model.py
```

This trains a logistic regression model and saves a classification report and confusion matrix.

### 3) Run the full modeling pipeline

```bash
python model_pipeline.py
```

This trains multiple models, compares their results, and creates customer segmentation output.

### 4) Open the Streamlit dashboard

```bash
python -m streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually http://localhost:8501.

## Project goal

The goal of this project is to help identify customers who may leave, understand the reasons behind churn, and support better retention decisions.
