# Telco Churn — Chunk Prediction Project

Files added:
- `eda.py` — runs exploratory data analysis and saves plots to `outputs/eda_plots`.
- `requirements.txt` — Python dependencies.

How to run EDA:

1. Create a virtual environment (optional):

```bash
python3 -m venv env
source env/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the EDA script from the repository root:

```bash
python eda.py
```

Outputs (PNG files) will be placed in `outputs/eda_plots`.

Streamlit dashboard

To run the interactive dashboard:

```bash
python -m streamlit run streamlit_app.py
```

Then open the URL shown by Streamlit (usually http://localhost:8501).
