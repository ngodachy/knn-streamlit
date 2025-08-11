

# streamlit_app.py
import io
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

if st.button("🔄 Reset App"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.experimental_rerun()



st.set_page_config(page_title="KNN – Streamlit", page_icon="🤖", layout="wide")
st.title("KNN Demo (Streamlit)")

# -------------------------------
# Helpers
# -------------------------------
SAMPLES = {
    "Breast Cancer": ("classification", load_breast_cancer(as_frame=True)),
    
}

DISTANCE_CHOICES = {
    "Euclidean (p=2)": ("minkowski", 2),
    "Manhattan (p=1)": ("minkowski", 1),
    "Minkowski (p=3)": ("minkowski", 3),
}

def load_sample(name: str):
    problem_type, bundle = SAMPLES[name]
    X = bundle.data.copy()
    y = bundle.target.copy()
    df = X.copy()
    df["target"] = y
    return df, "target", problem_type

@st.cache_data
def load_csv(file, sep=","):
    return pd.read_csv(file, sep=sep)

def validate_config(df, target_col, problem_type):
    if target_col not in df.columns:
        return False, "Please select a valid target column."
    target = df[target_col]
    if problem_type == "classification":
        if target.nunique() > 50:
            return False, f"Too many classes ({target.nunique()})."
        if target.isnull().any():
            return False, "Target contains missing values."
    elif problem_type == "regression":
        if target.dtype == "object":
            return False, "Text target detected. Use classification."
        if target.nunique() < 5:
            return False, f"Too few unique target values ({target.nunique()})."
    else:
        return False, "Pick problem type."
    return True, "OK"

def split_features(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols

def build_model(problem_type, k, metric_tuple, weighting, num_cols, cat_cols):
    metric, p = metric_tuple
    # Preprocess: scale numeric, one-hot encode categorical
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    if problem_type == "classification":
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, p=p, weights=weighting)
    else:
        model = KNeighborsRegressor(n_neighbors=k, metric=metric, p=p, weights=weighting)
    pipe = Pipeline([("pre", pre), ("knn", model)])
    return pipe

def default_value(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        return float(series.mean())
    # categorical: pick most frequent
    return series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else ""

def make_feature_inputs(df, target_col):
    X = df.drop(columns=[target_col])
    inputs = {}
    st.subheader(" Feature Inputs")
    cols = st.columns(2)
    for i, col in enumerate(X.columns):
        s = X[col]
        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(s):
                val = st.number_input(
                    f"{col}",
                    value=float(default_value(s)),
                    min_value=float(s.min()) if np.isfinite(s.min()) else None,
                    max_value=float(s.max()) if np.isfinite(s.max()) else None,
                    help="Numeric feature",
                )
            else:
                choices = sorted([str(v) for v in s.dropna().unique().tolist()]) or [""]
                val = st.selectbox(f"{col}", choices, index=0, help="Categorical feature")
            inputs[col] = val
    return inputs

def pca_plot(X_proc, labels, query_vec=None, title="PCA (2D)"):
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_proc)
    df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])
    if labels is not None:
        df_plot["label"] = labels
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="label" if labels is not None else None,
                     title=title, opacity=0.75)
    if query_vec is not None:
        q2 = pca.transform(query_vec)
        fig.add_scatter(x=[q2[0, 0]], y=[q2[0, 1]], mode="markers", marker_symbol="x",
                        marker_size=14, name="Query")
    return fig

# -------------------------------
# Sidebar: data source
# -------------------------------
st.sidebar.header(" Data")
mode = st.sidebar.radio("Choose data source", ["Sample dataset", "Upload CSV"], index=0)

if mode == "Sample dataset":
    sample_name = st.sidebar.selectbox("Dataset", list(SAMPLES.keys()), index=0)
    df, target_col_default, inferred_problem = load_sample(sample_name)
    problem_type = st.sidebar.selectbox("Problem type", ["classification", "regression"],
                                        index=0 if inferred_problem == "classification" else 1)
    target_col = st.sidebar.selectbox("Target column", options=df.columns, index=list(df.columns).index("target"))
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    sep = st.sidebar.text_input("CSV separator", value=",")
    df = load_csv(up, sep) if up else pd.DataFrame()
    st.sidebar.caption("After uploading, pick target/problem type:")
    target_col = st.sidebar.selectbox("Target column", options=df.columns if not df.empty else [])
    problem_type = st.sidebar.selectbox("Problem type", ["classification", "regression"])

# Quick preview
st.subheader(" Data Preview")
if df.empty:
    st.info("Upload a CSV or pick a sample dataset.")
    st.stop()
st.dataframe(df.head(), use_container_width=True)

# Validate config
ok, msg = validate_config(df, target_col, problem_type)
if not ok:
    st.error(msg)
    st.stop()
else:
    st.success(f" Config OK – {problem_type.capitalize()} on target **{target_col}**")

# -------------------------------
# Sidebar: KNN params
# -------------------------------
st.sidebar.header(" KNN Parameters")
k = st.sidebar.slider("K (neighbors)", min_value=1, max_value=50, value=3, step=1)
distance_label = st.sidebar.selectbox("Distance", list(DISTANCE_CHOICES.keys()), index=0)
weighting = st.sidebar.selectbox("Weighting", ["uniform", "distance"], index=0)

# -------------------------------
# Build/train model
# -------------------------------
X, y, num_cols, cat_cols = split_features(df, target_col)
pipe = build_model(problem_type, k, DISTANCE_CHOICES[distance_label], weighting, num_cols, cat_cols)
pipe.fit(X, y)

# We also keep a preprocessed design matrix for PCA viz and neighbor lookup
pre = pipe.named_steps["pre"]
X_proc = pre.fit_transform(X)  # fit_transform again to get an explicit matrix for PCA
if hasattr(X_proc, "toarray"):
    X_proc_dense = X_proc.toarray()
else:
    X_proc_dense = np.asarray(X_proc)

# -------------------------------
# Inputs & prediction
# -------------------------------
inputs = make_feature_inputs(df, target_col)

# Build a single-row DataFrame from inputs
query_df = pd.DataFrame([inputs])
pred = pipe.predict(query_df)[0]

st.subheader(" Prediction")
if problem_type == "classification":
    st.success(f"**Predicted class:** {pred}")
else:
    st.success(f"**Predicted value:** {pred:.4f}")

# -------------------------------
# Neighbors (table)
# -------------------------------
st.subheader(" Nearest Neighbors")
# Use NearestNeighbors on the same metric to show distances/indices
metric, p = DISTANCE_CHOICES[distance_label]
nn = NearestNeighbors(n_neighbors=min(k, len(X)), metric=metric, p=p)
nn.fit(X_proc_dense)
q_proc = pre.transform(query_df)
q_dense = q_proc.toarray() if hasattr(q_proc, "toarray") else np.asarray(q_proc)
dist, idx = nn.kneighbors(q_dense, return_distance=True)
neighbors = df.iloc[idx[0]].copy()
neighbors.insert(0, "distance", dist[0])
st.dataframe(neighbors.reset_index(drop=True), use_container_width=True)

# -------------------------------
# Visualization (PCA 2D)
# -------------------------------
st.subheader(" PCA Visualization")
labels = y if problem_type == "classification" else None
fig = pca_plot(X_proc_dense, labels, query_vec=q_dense, title="PCA (2D) of Preprocessed Features")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Summary
# -------------------------------
st.subheader(" Model Summary")
st.markdown(
    f"""
- **Problem**: {problem_type.capitalize()}  
- **Target column**: `{target_col}`  
- **#Samples**: {len(df)} | **#Features**: {X.shape[1]} (numeric: {len(num_cols)}, categorical: {len(cat_cols)})  
- **K**: {k} | **Distance**: {distance_label} | **Weighting**: {weighting}  
"""
)
