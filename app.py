# ============================================================
#  Bladder Cancer Synchronous Distant Metastasis Risk Calculator
#  Streamlit web app (GBDT model)
# ============================================================


import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="Bladder Cancer DM Calculator", page_icon="🔬", layout="centered")

@st.cache_resource
def load_model():
    M = joblib.load("gbdt_model.joblib")
    return M["pipe"], M["features"]
pipe, FEAT = load_model()

# 每个变量:显示标签 -> 模型编码
OPTIONS = {
    "AGE.GROUP":        ("Age group", {"< 60":1, "60–69":2, "70–79":3, "≥ 80":4}),
    "PRIMARY.SITE":     ("Primary site", {"Anterosuperior (anterior wall/dome)":1, "Lateral/posterior wall":2,
                                          "Trigone/neck/ureteric orifice":3, "Overlapping":4, "Bladder, NOS":5}),
    "HISTOLOGICAL.TYPE":("Histological type", {"Urothelial":1, "Squamous":2, "Adenocarcinoma":3,
                                               "Small cell / neuroendocrine":4, "Other variant":5}),
    "GRADE":            ("Grade", {"High grade":1, "Low grade":2, "Unknown":3}),
    "T":                ("Clinical T stage", {"T1":1, "T2":2, "T3":3, "T4":4}),
    "N":                ("Clinical N stage", {"N0":0, "N1":1, "N2":2, "N3":3}),
    "TUMOR.SIZE.GROUP": ("Tumor size", {"≤ 2 cm":1, "2.1–4 cm":2, "4.1–6 cm":3, "> 6 cm":4}),
    "NUMBER.OF.TUMORS": ("Number of tumors", {"Solitary":1, "Multiple":2}),
}

st.title("Bladder Cancer Synchronous Distant Metastasis Calculator")
st.caption("Gradient Boosting Machine model · for research use only, not a substitute for clinical judgment")
st.markdown("---")

st.subheader("Enter patient characteristics")
vals = {}
c1, c2 = st.columns(2)
items = list(OPTIONS.items())
for i, (key, (label, mapping)) in enumerate(items):
    col = c1 if i % 2 == 0 else c2
    choice = col.selectbox(label, list(mapping.keys()), key=key)
    vals[key] = mapping[choice]

st.markdown("---")
if st.button("Calculate metastasis risk", type="primary", use_container_width=True):
    X = pd.DataFrame([vals])[FEAT]
    prob = float(pipe.predict_proba(X)[0, 1])
    pct = prob * 100
    st.metric("Predicted probability of synchronous distant metastasis", f"{pct:.1f}%")
    st.progress(min(prob, 1.0))
    if pct < 5:
        st.success(f"**Low risk** ({pct:.1f}%). Distant metastasis is unlikely.")
    elif pct < 20:
        st.warning(f"**Intermediate risk** ({pct:.1f}%). Consider thorough metastatic workup.")
    else:
        st.error(f"**High risk** ({pct:.1f}%). Comprehensive staging / metastatic workup recommended.")
    st.caption("Risk bands (5% / 20%) are illustrative — adjust to your clinical context.")

st.markdown("---")
st.caption("Model: GBDT trained on SEER invasive bladder cancer (training cohort 2010–2017). "
           "External validation AUC ≈ 0.84. Predictors: age, primary site, histology, grade, "
           "cT, cN, tumor size, number of tumors.")
