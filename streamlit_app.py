import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List, Optional

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Value & Churn Risk Intelligence (v3)",
    page_icon="📉",
    layout="wide"
)

# -----------------------------
# Robust Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()

MODEL_FILENAME = "churn_pipeline.joblib"

# Try multiple candidate locations (common real-world run patterns)
CANDIDATE_MODEL_PATHS = [
    APP_DIR / MODEL_FILENAME,                       # same folder as streamlit_app.py
    APP_DIR / "artifacts" / MODEL_FILENAME,         # artifacts/ under app folder (root layout)
    CWD / MODEL_FILENAME,                           # current working directory
    CWD / "artifacts" / MODEL_FILENAME,             # artifacts/ under current working directory
]

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model_from_candidates(candidates: List[Path]):
    for p in candidates:
        if p.exists():
            return joblib.load(p), p
    # If not found, raise a detailed error
    tried = "\n".join([f"- {str(p)}" for p in candidates])
    raise FileNotFoundError(
        "Model file not found.\n"
        f"App folder: {APP_DIR}\n"
        f"Current working dir: {CWD}\n"
        "Tried these paths:\n"
        f"{tried}\n\n"
        "Fix options:\n"
        "1) Put churn_pipeline.joblib into ./artifacts/\n"
        "2) Or put it in the same folder as streamlit_app.py\n"
        "3) Or run Streamlit from the project root: streamlit run streamlit_app.py"
    )

pipe, MODEL_PATH_USED = load_model_from_candidates(CANDIDATE_MODEL_PATHS)

# Try to read expected input schema (works if trained with a DataFrame)
try:
    FEATURE_NAMES = list(pipe.feature_names_in_)
except Exception:
    FEATURE_NAMES = None

# -----------------------------
# Business Mappings
# -----------------------------
FEATURE_TO_BUSINESS = {
    "tenure": "Short customer tenure",
    "MonthlyCharges": "High monthly charges",
    "TotalCharges": "Total charges level",
    "Contract": "Contract type",
    "PaymentMethod": "Payment method",
    "InternetService": "Internet service type",
    "TechSupport": "Tech support status",
    "OnlineSecurity": "Online security status",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "MultipleLines": "Multiple phone lines",
}

BUSINESS_RATIONALE = {
    "Short customer tenure": "Early-stage customers are more fragile and may churn before forming habits or perceiving value.",
    "High monthly charges": "Higher bills increase price sensitivity when perceived value or service quality is insufficient.",
    "Contract type": "Month-to-month plans have lower switching costs and typically higher churn volatility.",
    "Payment method": "Certain payment behaviors can correlate with churn risk or billing friction.",
    "Internet service type": "Premium services can backfire if reliability/support does not meet expectations.",
    "Tech support status": "Lack of support can increase frustration during issues and reduce retention.",
    "Online security status": "Security add-ons can reflect perceived need/value; absence may correlate with lower stickiness.",
    "Multiple phone lines": "More lines can increase service complexity and potential support friction.",
}

RECOMMENDED_ACTIONS = {
    "Short customer tenure": [
        "Run a first-30-days onboarding/activation play (welcome + setup + usage nudges).",
        "Send a guided 'getting value fast' checklist and proactive check-in.",
    ],
    "High monthly charges": [
        "Offer plan right-sizing or a value-based bundle discount.",
        "Provide a loyalty credit tied to engagement or satisfaction survey completion.",
    ],
    "Contract type": [
        "Offer contract migration incentive (12–24 months) to reduce churn volatility.",
        "Position a 'price lock' benefit to reduce churn temptation.",
    ],
    "Internet service type": [
        "Proactive QoS monitoring + support outreach (latency/outages).",
        "Provide reliability tips and a clear escalation path.",
    ],
    "Payment method": [
        "Reduce billing friction: autopay enrollment incentive and reminders.",
        "Offer payment flexibility where appropriate.",
    ],
}

# -----------------------------
# Helpers
# -----------------------------
def risk_bucket(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"

def urgency_level(p: float) -> str:
    if p >= 0.80:
        return "Very High"
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"

def build_input_df(inputs: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame for prediction and align to training schema.
    Key goal: never pass NaN into the estimator.
    """
    df = pd.DataFrame([inputs])

    if FEATURE_NAMES is not None:
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = ""  # safe default for categoricals
        df = df[FEATURE_NAMES]

    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df[obj_cols] = df[obj_cols].fillna("")

    return df

def get_preprocessor_and_estimator(pipeline):
    """
    Best-effort: find a ColumnTransformer-like preprocessor and final estimator.
    """
    preprocessor = None
    estimator = None

    if hasattr(pipeline, "named_steps"):
        for k in ["preprocess", "preprocessor", "prep", "transformer"]:
            if k in pipeline.named_steps:
                preprocessor = pipeline.named_steps[k]
                break
        try:
            estimator = list(pipeline.named_steps.values())[-1]
        except Exception:
            estimator = None

    return preprocessor, estimator

def try_get_transformed_feature_names(preprocessor) -> Optional[List[str]]:
    if preprocessor is None:
        return None
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return None

def explain_top_drivers_linear(pipe, X_row_df: pd.DataFrame, top_n: int = 10) -> Optional[pd.DataFrame]:
    """
    Coefficient-based explanation for linear models:
    contribution = coef * transformed_value
    """
    preprocessor, estimator = get_preprocessor_and_estimator(pipe)
    if estimator is None or not hasattr(estimator, "coef_"):
        return None

    try:
        if preprocessor is not None:
            X_trans = preprocessor.transform(X_row_df)
            feat_names = try_get_transformed_feature_names(preprocessor)
        else:
            X_trans = X_row_df.values
            feat_names = list(X_row_df.columns)
    except Exception:
        return None

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    X_trans = np.asarray(X_trans).reshape(1, -1)
    coefs = np.asarray(estimator.coef_).reshape(-1)

    if X_trans.shape[1] != coefs.shape[0]:
        return None

    contrib = X_trans[0] * coefs
    if feat_names is None or len(feat_names) != len(contrib):
        feat_names = [f"f_{i}" for i in range(len(contrib))]

    out = pd.DataFrame({"feature": feat_names, "contribution": contrib})
    out["direction"] = np.where(out["contribution"] >= 0, "↑ increases churn risk", "↓ reduces churn risk")
    out["abs_contrib"] = out["contribution"].abs()
    out = out.sort_values("abs_contrib", ascending=False).head(top_n).drop(columns=["abs_contrib"])
    return out.reset_index(drop=True)

def map_to_business_driver(raw_feature: str) -> str:
    """
    Convert transformed feature name into a business-friendly label.
    Example: cat__Contract_Month-to-month -> Contract type (Month-to-month)
    """
    if "__" in raw_feature:
        raw_feature = raw_feature.split("__", 1)[1]

    base = raw_feature.split("_")[0] if "_" in raw_feature else raw_feature

    if base in FEATURE_TO_BUSINESS:
        if base in ["Contract", "PaymentMethod", "InternetService"]:
            cat = raw_feature.replace(base + "_", "")
            return f"{FEATURE_TO_BUSINESS[base]} ({cat})"
        return FEATURE_TO_BUSINESS[base]

    return raw_feature

def build_retention_playbook(top_drivers_df: pd.DataFrame) -> pd.DataFrame:
    out = top_drivers_df.copy()
    out["business_driver"] = out["feature"].apply(map_to_business_driver)

    def rationale(bd: str) -> str:
        for k, v in BUSINESS_RATIONALE.items():
            if bd.startswith(k):
                return v
        return "This signal is associated with churn risk based on model contribution."

    def actions(bd: str) -> str:
        for k, v in RECOMMENDED_ACTIONS.items():
            if bd.startswith(k):
                return "\n".join([f"- {a}" for a in v])
        return "- Review customer journey friction and offer a targeted retention touchpoint."

    out["business_rationale"] = out["business_driver"].apply(rationale)
    out["recommended_actions"] = out["business_driver"].apply(actions)

    out = out[["business_driver", "direction", "business_rationale", "recommended_actions", "contribution"]]
    out = out.rename(columns={"contribution": "model_contribution"})
    return out

def render_model_status():
    with st.expander("📦 Model & File Status", expanded=False):
        st.write("**App folder:**", str(APP_DIR))
        st.write("**Current working dir:**", str(CWD))
        st.write("**Model file used:**", str(MODEL_PATH_USED))
        st.write("**Model exists:**", Path(MODEL_PATH_USED).exists())

        st.write("**Tried candidate paths:**")
        st.code("\n".join([str(p) for p in CANDIDATE_MODEL_PATHS]))

        if FEATURE_NAMES is not None:
            st.write("**Expected input columns (from training):**")
            st.code(", ".join(FEATURE_NAMES))
        else:
            st.warning("feature_names_in_ not available (pipeline may have been trained on a NumPy array).")

def make_markdown_report(
    proba: float,
    bucket: str,
    urgency: str,
    top_drivers: Optional[pd.DataFrame],
    playbook: Optional[pd.DataFrame],
) -> str:
    lines = []
    lines.append("# Customer Churn Risk Report\n")
    lines.append(f"- **Churn Risk:** {proba*100:.2f}%")
    lines.append(f"- **Risk Level:** {bucket}")
    lines.append(f"- **Action Urgency:** {urgency}\n")

    if playbook is not None and not playbook.empty:
        lines.append("## Retention Playbook (Top Drivers)\n")
        for _, r in playbook.iterrows():
            lines.append(f"### {r['business_driver']}")
            lines.append(f"- **Direction:** {r['direction']}")
            lines.append(f"- **Why it matters:** {r['business_rationale']}")
            lines.append("**Recommended actions:**")
            lines.append(r["recommended_actions"])
            lines.append("")
    elif top_drivers is not None and not top_drivers.empty:
        lines.append("## Top Drivers (Model-based)\n")
        lines.append(top_drivers.to_markdown(index=False))
        lines.append("\n_Note: A richer playbook needs mapping drivers to business actions._")
    else:
        lines.append("## Top Drivers\n\n_Not available for this model/pipeline._")

    lines.append("\n---\n**Disclaimer:** Demo analytics tool (not production scoring).")
    return "\n".join(lines)

# -----------------------------
# Header
# -----------------------------
st.title("📉 Customer Value & Churn Risk Intelligence (v3)")
st.caption("Consultant-style demo: estimate churn risk, explain top drivers, generate retention actions, and score customers in batch.")
render_model_status()

tab1, tab2, tab3 = st.tabs([
    "🧍 Single Customer Scoring",
    "📄 Batch Scoring (CSV)",
    "🧠 About (Consultant Lens)",
])

# -----------------------------
# Tab 1: Single Customer
# -----------------------------
with tab1:
    left, right = st.columns([1.1, 1.5])

    with left:
        st.subheader("🧾 Customer Inputs")

        gender = st.selectbox("gender", ["Female", "Male"])
        senior = st.selectbox("SeniorCitizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        tenure = st.slider("tenure (months)", 0, 72, 12)
        monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=300.0, value=75.0, step=1.0)
        default_total = float(monthly * max(tenure, 1))
        total = st.number_input("TotalCharges", min_value=0.0, max_value=20000.0, value=default_total, step=10.0)

        phone = st.selectbox("PhoneService", ["Yes", "No"])
        multiple = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
        device_prot = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
        tech = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
        pay = st.selectbox("PaymentMethod", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ])

        run = st.button("🔍 Evaluate Churn Risk", use_container_width=True)

    with right:
        st.subheader("📌 Results")

        if run:
            inputs = {
                "gender": gender,
                "SeniorCitizen": int(senior),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": int(tenure),
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": online_sec,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_prot,
                "TechSupport": tech,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": pay,
                "MonthlyCharges": float(monthly),
                "TotalCharges": float(total),
            }

            X = build_input_df(inputs)

            try:
                proba = float(pipe.predict_proba(X)[0][1])
                bucket = risk_bucket(proba)
                urgency = urgency_level(proba)

                c1, c2, c3 = st.columns(3)
                c1.metric("Churn Risk", f"{proba*100:.2f}%")
                c2.metric("Risk Level", bucket)
                c3.metric("Action Urgency", urgency)

                st.markdown("### 🔎 Top Drivers (if supported)")
                top_drivers = explain_top_drivers_linear(pipe, X, top_n=10)

                playbook = None
                if top_drivers is not None and not top_drivers.empty:
                    st.dataframe(top_drivers, use_container_width=True)

                    playbook = build_retention_playbook(top_drivers)
                    st.markdown("### ✅ Retention Playbook (Consultant-style)")
                    show_cols = ["business_driver", "direction", "business_rationale", "recommended_actions"]
                    st.dataframe(playbook[show_cols], use_container_width=True)
                else:
                    st.info("Top-driver explanations are not available for this pipeline configuration.")

                report_md = make_markdown_report(proba, bucket, urgency, top_drivers, playbook)
                st.download_button(
                    "⬇️ Download Customer Risk Report (Markdown)",
                    data=report_md.encode("utf-8"),
                    file_name="customer_churn_risk_report.md",
                    mime="text/markdown",
                )

                with st.expander("🧾 View model input row"):
                    st.dataframe(X, use_container_width=True)

            except Exception as e:
                st.error("Prediction failed. This usually means schema mismatch or missing preprocessing in the saved pipeline.")
                st.exception(e)
                st.info("Tip: Re-save the pipeline from Notebook 3 and ensure the joblib includes preprocessing + model.")

        else:
            st.info("Fill inputs on the left, then click **Evaluate Churn Risk**.")

# -----------------------------
# Tab 2: Batch Scoring
# -----------------------------
with tab2:
    st.subheader("📄 Batch Scoring (CSV Upload)")
    st.write("Upload a CSV with the Telco churn schema. The app outputs churn risk and risk bucket per row.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df_in.head(10), use_container_width=True)

            if FEATURE_NAMES is not None:
                missing_cols = [c for c in FEATURE_NAMES if c not in df_in.columns]
                if missing_cols:
                    st.warning(f"Missing columns filled with defaults: {missing_cols}")
                    for c in missing_cols:
                        df_in[c] = ""
                df_scoring = df_in[FEATURE_NAMES].copy()
            else:
                df_scoring = df_in.copy()
                st.warning("Training schema not found (feature_names_in_ missing). Using uploaded columns.")

            for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
                if col in df_scoring.columns:
                    df_scoring[col] = pd.to_numeric(df_scoring[col], errors="coerce").fillna(0)

            obj_cols = df_scoring.select_dtypes(include=["object"]).columns.tolist()
            if obj_cols:
                df_scoring[obj_cols] = df_scoring[obj_cols].fillna("")

            proba = pipe.predict_proba(df_scoring)[:, 1]
            out = df_in.copy()
            out["churn_risk_pct"] = np.round(proba * 100, 2)
            out["risk_level"] = [risk_bucket(p) for p in proba]
            out["action_urgency"] = [urgency_level(p) for p in proba]

            st.success("Batch scoring completed.")
            st.dataframe(out.head(30), use_container_width=True)

            st.download_button(
                "⬇️ Download Scored CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="batch_scored_churn_risk.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("Batch scoring failed. Please verify your CSV schema matches training data.")
            st.exception(e)
    else:
        st.info("Upload a CSV to score churn risk in batch.")

# -----------------------------
# Tab 3: About
# -----------------------------
with tab3:
    st.subheader("🧠 Consultant Lens: How this becomes a client deliverable")
    st.markdown(
        """
**What this demo shows**
- A reusable churn scoring engine (saved model pipeline + Streamlit app)
- Explainable risk drivers (coefficient-based contributions)
- Actionable retention playbooks (driver → rationale → recommended actions)
- Batch scoring for campaign operations

**How you'd use it in a real engagement**
- Segment customers by risk level and design targeted retention plays
- Run A/B tests (offer types, contract incentives, onboarding sequences)
- Monitor drift via monthly scoring + stability checks

**Next upgrades (v4 ideas)**
- SHAP explanations for non-linear models (XGBoost/LightGBM)
- Cost-sensitive thresholding (optimize for retention ROI)
- Campaign simulator: expected saves vs discount cost
        """
    )

st.caption("v3 note: Top Drivers is coefficient-based and works best with Logistic Regression pipelines.")