import os
import json
import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import shap
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config and upgraded styling
st.set_page_config(page_title='Telco Churn Dashboard', layout='wide')
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', 'Segoe UI', Arial, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #F8FAFB 0%, #FFFFFF 50%, #F0F4F8 100%);
    }

    h1, h2, h3 {
        color: #1E3A5F !important;
        letter-spacing: 0.5px;
    }

    p, span, label, div {
        color: #2D3E50 !important;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1.2rem;
    }

    .stButton>button {
        background: linear-gradient(90deg, #1E40AF, #1e3a8a);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1rem;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #2563EB, #1e40af);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EEF2F8 0%, #E8EDF3 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #2D3E50 !important;
    }

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1E40AF !important;
        text-transform: uppercase;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: #1E40AF !important;
        font-weight: 800;
        font-size: 1.5em;
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stCaption {
        color: #315C70 !important;
    }

    [data-baseweb="select"] input {
        color: #083B53 !important;
    }

    .stCaption {
        color: #64748B !important;
    }
    .metric-revenue { color: #059669 !important; }
    .metric-churn { color: #DC2626 !important; }
    .metric-customers { color: #0064A3 !important; }
    .metric-contract { color: #7C3AED !important; }
    .metric-services { color: #F59E0B !important; }    </style>
    """,
    unsafe_allow_html=True,
)
from model_pipeline import load_and_clean, feature_engineering


CHART_COLORS = ["#1ECC07", '#059669', '#DC2626', '#7C3AED', '#F59E0B', '#06B6D4', '#EC4899', '#14B8A6']


def style_fig(fig, height=360):
    fig.update_layout(
        template='plotly_white',
        colorway=CHART_COLORS,
        font=dict(family='Manrope, Segoe UI, Arial', size=13, color='#2D3E50'),
        margin=dict(l=20, r=20, t=56, b=26),
        height=height,
        legend_title_text='',
        paper_bgcolor="#F8F8F8",
        plot_bgcolor="#FFFFFF",
        title_font=dict(size=18, color='#1E40AF'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(30, 64, 175, 0.1)', gridwidth=0.5)
    fig.update_yaxes(gridcolor='rgba(40, 74, 275, 0.1)', zeroline=False, gridwidth=0.5)
    return fig


def app_title():
    st.markdown(
        """
        <div style='text-align:center; padding: 1.5rem 0rem 2rem 0rem;'>
            <h1 style='margin-bottom:0.3rem; font-size: 2.8em; letter-spacing: 2px; color: #1E3A5F;'>🎯 CHURN ANALYTICS</h1>
            <p style='color:#1E40AF; margin-top:0; font-size: 1.1em; letter-spacing: 1px; text-transform: uppercase;'>Predictive Insights & Customer Risk Segmentation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_data(df):
    st.markdown("### Dataset Sample")
    st.caption('Preview of first 10 records')
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')
    st.download_button('Download Full Dataset CSV', data=df.to_csv(index=False), file_name='dataset.csv')


def show_eda_images():
    eda_dir = os.path.join('outputs', 'eda_plots')
    if os.path.exists(eda_dir):
        imgs = sorted([f for f in os.listdir(eda_dir) if f.lower().endswith('.png')])
        cols = st.columns(2)
        for i, img in enumerate(imgs):
            with cols[i % 2]:
                st.image(Image.open(os.path.join(eda_dir, img)), caption=img)
    else:
        st.write('No EDA images found. Run `python eda.py` to generate plots.')


def plot_churn_pie(df):
    counts = df['Churn'].value_counts().reset_index()
    counts.columns = ['Churn', 'count']
    churn_colors = {'Yes': '#DC2626', 'No': "#279605"}
    fig = px.pie(counts, names='Churn', values='count', hole=0.45, title='Churn Distribution',
                 color='Churn', color_discrete_map=churn_colors)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return style_fig(fig)


def plot_revenue_trend(df):
    # monthly revenue by tenure buckets
    df2 = df.groupby('tenure', as_index=False)['MonthlyCharges'].mean()
    fig = px.line(df2, x='tenure', y='MonthlyCharges', title='Average Monthly Charges by Tenure')
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(title='Tenure (months)')
    fig.update_yaxes(title='Avg Monthly Charges')
    return style_fig(fig)


def plot_contract_churn(df):
    d = df.groupby(['Contract', 'Churn'], as_index=False).size()
    churn_colors = {'Yes': '#DC2626', 'No': '#059669'}
    fig = px.bar(
        d,
        x='Contract',
        y='size',
        color='Churn',
        barmode='group',
        color_discrete_map=churn_colors,
        title='Churn by Contract Type',
    )
    fig.update_xaxes(title='Contract Type')
    fig.update_yaxes(title='Customer Count')
    return style_fig(fig)


def plot_monthly_box(df):
    churn_colors = {'Yes': '#DC2626', 'No': '#059669'}
    fig = px.box(
        df,
        x='Churn',
        y='MonthlyCharges',
        color='Churn',
        color_discrete_map=churn_colors,
        title='Monthly Charges Distribution by Churn',
    )
    fig.update_xaxes(title='Churn')
    fig.update_yaxes(title='Monthly Charges ($)')
    return style_fig(fig)


def plot_corr_heatmap(df):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    corr = num.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis'))
    fig.update_layout(title='Correlation Heatmap')
    return style_fig(fig, height=500)


def plot_segmentation_pie(seg_path):
    if not os.path.exists(seg_path):
        return None
    seg = pd.read_csv(seg_path)
    counts = seg['risk_category'].value_counts().reset_index()
    counts.columns = ['risk', 'count']
    risk_colors = {'Low': '#059669', 'Medium': '#F59E0B', 'High': '#DC2626'}
    fig = px.pie(counts, names='risk', values='count', hole=0.5, title='Risk Category Distribution',
                 color='risk', color_discrete_map=risk_colors)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return style_fig(fig)


def plot_total_charges_box(df):
    """Box plot of Total Charges by Churn status"""
    churn_colors = {'Yes': '#DC2626', 'No': '#059669'}
    fig = px.box(
        df,
        x='Churn',
        y='TotalCharges',
        color='Churn',
        color_discrete_map=churn_colors,
        title='Total Charges Distribution by Churn',
    )
    fig.update_xaxes(title='Churn')
    fig.update_yaxes(title='Total Charges ($)')
    return style_fig(fig)


def plot_internet_service_churn(df):
    """Internet Service type vs Churn"""
    d = df.groupby(['InternetService', 'Churn'], as_index=False).size()
    churn_colors = {'Yes': '#DC2626', 'No': '#059669'}
    fig = px.bar(
        d,
        x='InternetService',
        y='size',
        color='Churn',
        barmode='group',
        color_discrete_map=churn_colors,
        title='Churn by Internet Service Type',
    )
    fig.update_xaxes(title='Internet Service')
    fig.update_yaxes(title='Customer Count')
    return style_fig(fig)


def plot_tenure_histogram(df):
    """Tenure distribution with churn comparison"""
    churn_colors = {'Yes': '#DC2626', 'No': '#059669'}
    fig = px.histogram(
        df,
        x='tenure',
        color='Churn',
        nbins=30,
        color_discrete_map=churn_colors,
        title='Customer Tenure Distribution by Churn',
        labels={'tenure': 'Tenure (months)', 'count': 'Number of Customers'}
    )
    fig.update_xaxes(title='Tenure (months)')
    fig.update_yaxes(title='Number of Customers')
    return style_fig(fig)


def plot_services_count(df):
    """Number of services distribution"""
    if 'num_services' not in df.columns:
        return None
    services_churn = df.groupby(['num_services', 'Churn'], as_index=False).size()
    churn_colors = {'Yes': '#DC2626', 'No': '#059669'}
    fig = px.bar(
        services_churn,
        x='num_services',
        y='size',
        color='Churn',
        barmode='group',
        color_discrete_map=churn_colors,
        title='Customers by Number of Services & Churn',
    )
    fig.update_xaxes(title='Number of Services')
    fig.update_yaxes(title='Customer Count')
    return style_fig(fig)


def show_models_comparison():
    """Display all 5 trained models performance metrics side-by-side"""
    summary_path = os.path.join('outputs', 'models', 'summary_metrics.json')
    if not os.path.exists(summary_path):
        st.warning('Model metrics not found. Run `python model_pipeline.py` first.')
        return
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    models = summary.get('models', {})
    best = summary.get('best_model', 'gradient_boosting')
    
    st.markdown('### 🔬 Model Performance Comparison')
    
    # Create comparison dataframe
    comp_data = []
    for model_name, metrics in models.items():
        comp_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
        })
    
    df_comp = pd.DataFrame(comp_data)
    df_comp = df_comp.sort_values('ROC-AUC', ascending=False)
    
    # Display table
    st.dataframe(df_comp, use_container_width=True)
    
    # ROC-AUC bar chart
    try:
        fig = px.bar(df_comp, x='Model', y='ROC-AUC', color='ROC-AUC', title='Model ROC-AUC Comparison')
        fig = style_fig(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # Highlight best model
    best_display = best.replace('_', ' ').title()
    best_metrics = models.get(best, {})
    st.markdown(f"**🏆 Best Model:** {best_display}")
    st.markdown(f"- **ROC-AUC:** {best_metrics.get('roc_auc', 0):.4f}")
    st.markdown(f"- **Accuracy:** {best_metrics.get('accuracy', 0):.4f}")
    st.markdown(f"- **F1-Score:** {best_metrics.get('f1', 0):.4f}")


def load_best_model():
    summary_path = os.path.join('outputs', 'models', 'summary_metrics.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        best = summary.get('best_model', 'gradient_boosting')
    else:
        best = 'gradient_boosting'

    # Map model names to file names
    model_map = {
        'logistic': 'logistic_regression.joblib',
        'decision_tree': 'decision_tree.joblib',
        'random_forest': 'random_forest.joblib',
        'gradient_boosting': 'gradient_boosting.joblib',
        'svm': 'svm.joblib',
    }
    
    model_file = model_map.get(best, 'gradient_boosting.joblib')
    model_path = os.path.join('outputs', 'models', model_file)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model, best
    return None, None


def build_explainer(model, df_raw):
    # prepare background dataset for SHAP
    try:
        df_fe = feature_engineering(df_raw)
        X_bg = pd.get_dummies(df_fe, drop_first=True)
        # align to model feature names if available
        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
            X_bg = X_bg.reindex(columns=expected, fill_value=0)
        else:
            feat_path = os.path.join('outputs', 'models', 'feature_columns.json')
            if os.path.exists(feat_path):
                with open(feat_path) as f:
                    expected = json.load(f)
                X_bg = X_bg.reindex(columns=expected, fill_value=0)
        # sample background to reasonable size
        bg = X_bg.sample(min(100, len(X_bg)), random_state=1)
        explainer = shap.Explainer(model, bg, feature_names=bg.columns.tolist())
        return explainer, bg
    except Exception as e:
        st.warning('SHAP explainer could not be built: ' + str(e))
        return None, None


def compute_global_shap_importance(model, df_raw, top_k=15):
    explainer, bg = build_explainer(model, df_raw)
    if explainer is None:
        return None
    try:
        # compute SHAP values on background sample
        shap_vals = explainer(bg)
        vals = np.abs(shap_vals.values).mean(axis=0)
        feat_names = shap_vals.feature_names
        imp = pd.DataFrame({'feature': feat_names, 'importance': vals})
        imp = imp.sort_values('importance', ascending=False).head(top_k)
        return imp
    except Exception as e:
        st.warning('Could not compute global SHAP importance: ' + str(e))
        return None


def predict_dataframe(model, df, best_model_name='gradient_boosting'):
    # apply same feature engineering used in pipeline
    df_fe = feature_engineering(df)
    X = pd.get_dummies(df_fe, drop_first=True)

    # Align columns to model expected feature names
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        X = X.reindex(columns=expected, fill_value=0)
    else:
        # fallback: try to load saved feature columns
        feat_path = os.path.join('outputs', 'models', 'feature_columns.json')
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                expected = json.load(f)
            X = X.reindex(columns=expected, fill_value=0)
        else:
            st.warning('Model feature names not available; predictions may fail if columns mismatch.')

    # Handle SVM which requires scaling
    if best_model_name == 'svm':
        scaler_path = os.path.join('outputs', 'models', 'svm_scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)

    probs = model.predict_proba(X)[:,1]
    df_out = df.copy()
    df_out['churn_prob'] = probs
    df_out['risk_category'] = df_out['churn_prob'].apply(lambda p: 'High' if p>=0.66 else ('Medium' if p>=0.33 else 'Low'))
    return df_out


def train_and_show():
    st.info('Training through the web UI is disabled here; use the Train Model tab to run training script locally.')


def main():
    app_title()
    st.sidebar.markdown('### Navigation')

    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(csv_path):
        st.error(f'CSV not found at {csv_path}. Run from project root.')
        return

    df = load_and_clean(csv_path)

    # attempt to read saved summary metrics
    summary_path = os.path.join('outputs', 'models', 'summary_metrics.json')
    best_metrics = None
    best_name = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            best_name = summary.get('best_model')
            best_metrics = summary.get(best_name)
        except Exception:
            best_metrics = None

    section = st.sidebar.selectbox('Choose view', ['Overview', 'Data', 'EDA Plots', 'Predict'])

    if section == 'Overview':
        st.subheader('📊 Business Metrics')
        
        # Calculate business metrics
        total_customers = len(df)
        churned = (df['Churn'] == 'Yes').sum()
        churn_rate = (churned / total_customers * 100) if total_customers > 0 else 0
        avg_monthly = df['MonthlyCharges'].mean()
        total_revenue = df['TotalCharges'].sum()
        avg_tenure = df['tenure'].mean()
        
        # Get high-risk customers
        seg_path = os.path.join('outputs', 'segmentation.csv')
        high_risk = 0
        if os.path.exists(seg_path):
            seg = pd.read_csv(seg_path)
            high_risk = (seg['risk_category'] == 'High').sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('👥 Total Customers', f"{total_customers:,}")
        c2.metric('📉 Churn Rate', f"{churn_rate:.1f}%")
        c3.metric('💰 Avg Monthly Charges', f"${avg_monthly:.2f}")
        c4.metric('📅 Avg Tenure', f"{avg_tenure:.1f} mo")

        c5, c6, c7 = st.columns(3)
        c5.metric('⚠️ Churned Customers', f"{churned:,}")
        c6.metric('🔴 High-Risk Customers', f"{high_risk:,}")
        c7.metric('💵 Total Revenue', f"${total_revenue:,.0f}")

        st.markdown('---')
        show_models_comparison()
        st.markdown('---')
        st.subheader('📋 Segmentation Summary')
        if os.path.exists(seg_path):
            seg = pd.read_csv(seg_path)
            st.write(seg['risk_category'].value_counts())
        else:
            st.info('Run `python model_pipeline.py` to generate segmentation.')

        # global SHAP importance
        model, _ = load_best_model()
        if model is not None:
            imp = compute_global_shap_importance(model, df, top_k=12)
            if imp is not None:
                st.subheader('🔍 Feature Importance (SHAP)')
                st.bar_chart(imp.set_index('feature').importance)
        # professional charts
        st.markdown('---')
        c1, c2 = st.columns([1,2])
        with c1:
            fig = plot_churn_pie(df)
            st.plotly_chart(fig, use_container_width=True)
            churn_counts = df['Churn'].value_counts().rename_axis('Churn').reset_index(name='count')
            st.download_button('Download churn chart data', data=churn_counts.to_csv(index=False), file_name='churn_distribution.csv')
        with c2:
            fig2 = plot_revenue_trend(df)
            st.plotly_chart(fig2, use_container_width=True)
            rev = df.groupby('tenure', as_index=False)['MonthlyCharges'].mean()
            st.download_button('Download revenue trend data', data=rev.to_csv(index=False), file_name='revenue_trend.csv')

    elif section == 'Data':
        show_data(df)
    elif section == 'EDA Plots':
        show_eda_images()
        st.markdown('---')
        st.subheader('Additional Interactive Charts')
        col1, col2 = st.columns(2)
        with col1:
            heat = plot_corr_heatmap(df)
            if heat is not None:
                st.plotly_chart(heat, use_container_width=True)
            else:
                st.info('Not enough numeric columns for correlation heatmap.')
        with col2:
            seg_path = os.path.join('outputs', 'segmentation.csv')
            seg_fig = plot_segmentation_pie(seg_path)
            if seg_fig is not None:
                st.plotly_chart(seg_fig, use_container_width=True)
                seg = pd.read_csv(seg_path)
                seg_counts = seg['risk_category'].value_counts().rename_axis('risk_category').reset_index(name='count')
                st.download_button('Download risk chart data', data=seg_counts.to_csv(index=False), file_name='risk_distribution.csv')
            else:
                st.info('Run pipeline to generate segmentation for pie chart.')

        st.markdown('---')
        st.subheader('Charges & Service Analysis')
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_contract_churn(df), use_container_width=True)
        with col4:
            st.plotly_chart(plot_monthly_box(df), use_container_width=True)

        st.markdown('---')
        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(plot_total_charges_box(df), use_container_width=True)
        with col6:
            st.plotly_chart(plot_internet_service_churn(df), use_container_width=True)

        st.markdown('---')
        col7, col8 = st.columns(2)
        with col7:
            st.plotly_chart(plot_tenure_histogram(df), use_container_width=True)
        with col8:
            services_fig = plot_services_count(df)
            if services_fig is not None:
                st.plotly_chart(services_fig, use_container_width=True)
            else:
                st.info('Services count chart not available.')
    elif section == 'Predict':
        model, best = load_best_model()
        if model is None:
            st.error('Trained model not found. Run the pipeline: `python model_pipeline.py`')
            return
        st.write(f'Loaded best model: {best}')

        # build SHAP explainer (best-effort)
        explainer, bg = build_explainer(model, df)

        # single customer lookup if available
        if 'customerID' in df.columns:
            st.subheader('Lookup a customer')
            sample_ids = df['customerID'].sample(200, random_state=1).tolist()
            choose = st.selectbox('Select customerID', [''] + sample_ids)
            if choose:
                cust = df[df['customerID'] == choose]
                if not cust.empty:
                    pred = predict_dataframe(model, cust, best)
                    st.table(pred[['customerID','churn_prob','risk_category']])
                    # show SHAP for this customer
                    if explainer is not None:
                        try:
                            # prepare aligned X
                            Xc = feature_engineering(cust)
                            Xc = pd.get_dummies(Xc, drop_first=True)
                            if hasattr(model, 'feature_names_in_'):
                                Xc = Xc.reindex(columns=list(model.feature_names_in_), fill_value=0)
                            shap_vals = explainer(Xc)
                            vals = shap_vals.values[0]
                            feat_names = shap_vals.feature_names
                            imp = pd.DataFrame({'feature': feat_names, 'shap_value': vals})
                            imp = imp.reindex(imp.shap_value.abs().sort_values(ascending=False).index)
                            st.subheader('Top SHAP contributions (this customer)')
                            st.bar_chart(imp.set_index('feature').shap_value.head(10))
                        except Exception as e:
                            st.warning('Could not compute SHAP for this customer: ' + str(e))

        st.markdown('---')
        upload = st.file_uploader('Upload CSV for prediction (same raw columns as training)', type=['csv'])
        if upload is not None:
            pred_df = pd.read_csv(upload)
            res = predict_dataframe(model, pred_df, best)
            st.subheader('Predictions')
            st.dataframe(res.head(50))
            st.download_button('Download predictions CSV', data=res.to_csv(index=False), file_name='predictions.csv')
        else:
            if st.button('Predict sample from dataset'):
                sample = df.sample(50, random_state=42)
                res = predict_dataframe(model, sample, best)
                st.subheader('Sample Predictions')
                st.dataframe(res.head(50))
                st.download_button('Download sample predictions', data=res.to_csv(index=False), file_name='sample_predictions.csv')


if __name__ == '__main__':
    main()
