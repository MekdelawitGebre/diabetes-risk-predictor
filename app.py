# app.py
"""
Hospital Diabetes Risk Dashboard - Full Production-Ready Version
Features:
- Home: KPIs, probability distribution, trends, recent predictions
- Dashboard: Single patient prediction with SHAP, gauge, threshold
- Model Insights: Feature importance, ROC, Precision-Recall, Calibration, Confusion Matrix
- Patient History: View & download past predictions
- Batch Upload: Upload CSV and get predictions

- Settings: Clear history, show DB info
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.calibration import calibration_curve
from src.data.preprocess import clean_data, preprocess_features

# --------------------------
# CONFIG & STYLING
# --------------------------
MODEL_PATH = "models/XGBoost_Best.pkl"
DB_PATH = "data/predictions_history.db"
MODEL_VERSION = "1.0"

st.set_page_config(
    page_title="Hospital Diabetes Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body { background-color: #f9fafb; font-family: 'Segoe UI', sans-serif; color:#111827; }
.big-title { font-size:32px; font-weight:700; color:#1e40af; margin-bottom:0; }
.sub { color:#374151; font-size:16px; margin-bottom:20px; }
.card { background:white; padding:18px; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.1); margin-bottom:15px; }
.metric-title { font-weight:600; color:#1e3a8a; }
.stButton>button { background-color: #1e40af; color:white; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# DATABASE
# --------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        pregnancies INTEGER,
        glucose REAL,
        bloodpressure REAL,
        skinthickness REAL,
        insulin REAL,
        bmi REAL,
        dpf REAL,
        age INTEGER,
        prediction INTEGER,
        probability REAL
    )""")
    conn.commit()
    conn.close()

def save_history(row):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    INSERT INTO history (timestamp,pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,dpf,age,prediction,probability)
    VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, row)
    conn.commit()
    conn.close()

def get_history(limit=500):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM history ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

init_db()

# --------------------------
# MODEL LOADING
# --------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    with st.spinner("⏳ Loading AI model..."):
        model = joblib.load(path)
    return model

@st.cache_resource
def load_shap_explainer(_model):
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return shap.Explainer(_model)

model = load_model()
explainer = load_shap_explainer(model)

def model_feature_names(m, sample_df):
    if isinstance(m, (xgb.XGBClassifier, xgb.XGBRegressor)):
        try:
            return m.get_booster().feature_names
        except Exception:
            return list(sample_df.columns)
    return list(sample_df.columns)

# --------------------------
# SIDEBAR
# --------------------------
page = st.sidebar.radio("Navigation", ["Home", "Dashboard", "Model Insights", "Patient History", "Batch Upload", "Settings"])
st.sidebar.markdown("---")
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, step=0.01)

# --------------------------
# HOME PAGE
# --------------------------
if page == "Home":
    st.markdown('<p class="big-title">🏥 Hospital Diabetes Risk Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Interactive overview of predictions and AI insights</p>', unsafe_allow_html=True)

    history = get_history(limit=1000)
    total = len(history)
    avg_prob = history['probability'].mean() if total>0 else 0
    positives = history[history['prediction']==1]
    pos_rate = len(positives)/total if total>0 else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Total Predictions", value=total)
    kpi2.metric(label="Average Risk", value=f"{avg_prob*100:.2f}%")
    kpi3.metric(label="Positive Rate", value=f"{pos_rate*100:.2f}%")

    st.markdown("---")

    if total>0:
    
        # Trend over time
        history['timestamp'] = pd.to_datetime(history['timestamp'])
        fig_time = px.line(history.sort_values('timestamp'), x='timestamp', y='probability', color='prediction',
                           color_discrete_map={0:'#22c55e',1:'#ef4444'},
                           title='Prediction Probabilities Over Time',
                           labels={'timestamp':'Timestamp','probability':'Probability','prediction':'Prediction'})
        st.plotly_chart(fig_time, use_container_width=True)

        # Recent predictions
        st.write("### Recent Predictions")
        st.dataframe(history.head(20), use_container_width=True)

# --------------------------
# DASHBOARD PAGE
# --------------------------
elif page == "Dashboard":
    st.header("🧍 Patient Risk Assessment")
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        Glucose = st.number_input("Glucose", 0, 300, 120)
        BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    with col2:
        SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
        Insulin = st.number_input("Insulin", 0, 900, 80)
        BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
    with col3:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        Age = st.number_input("Age", 0, 120, 35)

    if st.button("🩺 Predict Risk"):
        with st.spinner("🤖 AI is processing..."):
            df = pd.DataFrame([{"Pregnancies": Pregnancies, "Glucose": Glucose, "BloodPressure": BloodPressure,
                                "SkinThickness": SkinThickness, "Insulin": Insulin, "BMI": BMI,
                                "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age}])
            df_proc = preprocess_features(clean_data(df))
            X_input = df_proc.reindex(columns=model_feature_names(model, df_proc), fill_value=0)
            prediction = int(model.predict(X_input)[0])
            probability = float(model.predict_proba(X_input)[0][1])
            save_history((datetime.utcnow().isoformat(), Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age, prediction, probability))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "🟥 Diabetic" if probability>=threshold else "🟩 Non-Diabetic")
            st.metric("Probability", f"{probability*100:.2f}%")
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=probability*100,
                                               title={'text':"Diabetes Risk (%)"},
                                               gauge={'axis': {'range':[0,100]},
                                                      'bar': {'color':'#ef4444' if probability>0.7 else '#f59e0b' if probability>0.4 else '#22c55e'}}))
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col2:
            st.subheader("Feature Impact (SHAP)")
            try:
                shap_values = explainer(X_input)
                vals = shap_values.values[0] if shap_values.values.ndim>1 else shap_values.values
                shap_df = pd.DataFrame({'Feature': X_input.columns, 'SHAP Value': vals}).sort_values('SHAP Value')
                fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                                  color='SHAP Value', color_continuous_scale='RdBu')
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")

# --------------------------
# MODEL INSIGHTS
# --------------------------
elif page == "Model Insights":
    st.header("📊 Model Insights & Performance")

    uploaded = st.file_uploader("Upload CSV to compute validation metrics", type=["csv"])
    if uploaded:
        try:
            df_train = pd.read_csv(uploaded)
            X = preprocess_features(clean_data(df_train)).drop(columns=["Outcome"], errors="ignore")
            y = df_train["Outcome"]
            X = X.reindex(columns=model_feature_names(model, X), fill_value=0)
            y_prob = model.predict_proba(X)[:,1]

            # Feature importance
            importances = model.feature_importances_
            feat_names = model_feature_names(model, pd.DataFrame(columns=[]))
            fi = pd.Series(importances, index=feat_names).sort_values()
            fig_fi = px.bar(fi.tail(15), orientation='h', title="Top Feature Importances", color=fi.tail(15),
                            color_continuous_scale="Blues")
            st.plotly_chart(fig_fi, use_container_width=True)

            # ROC
            fpr, tpr, _ = roc_curve(y, y_prob)
            auc = roc_auc_score(y, y_prob)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC (AUC={auc:.2f})", line=dict(color='blue')))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
            roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(roc_fig, use_container_width=True)


            # Calibration
            prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
            calib_fig = go.Figure()
            calib_fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Calibration', line=dict(color='orange')))
            calib_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
            calib_fig.update_layout(title="Calibration Curve", xaxis_title="Predicted Probability", yaxis_title="Observed Probability")
            st.plotly_chart(calib_fig, use_container_width=True)




        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")
    else:
        st.info("Upload a CSV file to visualize validation metrics.")

# --------------------------
# PATIENT HISTORY
# --------------------------
elif page == "Patient History":
    st.header("📋 Patient Predictions History")
    hist = get_history()
    if hist.empty:
        st.info("No predictions yet.")
    else:
        st.dataframe(hist, use_container_width=True)
        st.download_button("Download History CSV", hist.to_csv(index=False).encode(), file_name="history.csv")

# --------------------------
# BATCH UPLOAD
# --------------------------
elif page == "Batch Upload":
    st.header("📁 Batch Prediction Upload")
    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Uploaded {len(df)} records.")
        with st.spinner("🤖 AI is processing batch..."):
            try:
                df_clean = clean_data(df)
                df_proc = preprocess_features(df_clean)
                X_all = df_proc.reindex(columns=model_feature_names(model, df_proc), fill_value=0)
                preds = model.predict(X_all)
                probs = model.predict_proba(X_all)[:,1]
                df["Prediction"] = preds
                df["Probability"] = probs
                st.success("Batch processed successfully.")
                st.dataframe(df.head(20))
                st.download_button("Download Results CSV", df.to_csv(index=False).encode(), file_name="batch_results.csv")
            except Exception as e:
                st.error(f"Batch processing failed: {e}")



# --------------------------
# SETTINGS
# --------------------------

elif page == "Settings":
    st.header("⚙️ Admin Settings")
    if st.button("Clear History Database"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        st.success("History cleared.")
    st.markdown(f"**Database Path:** `{DB_PATH}`")
