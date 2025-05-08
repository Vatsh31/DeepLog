import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# -----------------------
# Cleaning function
# -----------------------
def clean_log(text):
    text = text.lower()
    text = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', text)
    text = re.sub(r'[0-9a-f\-]{36}', '<UUID>', text)
    text = re.sub(r'\d+', '<NUM>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------
# Clustering function
# -----------------------
def cluster_logs(df, fast_mode):
    if fast_mode:
        df = df.drop_duplicates(subset='Cleaned')
        tfidf = TfidfVectorizer(max_features=1000)
    else:
        tfidf = TfidfVectorizer(max_features=3000)

    X = tfidf.fit_transform(df['Cleaned'])
    model = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    labels = model.fit_predict(X.toarray())
    df['Cluster'] = labels
    df['Is_Anomaly'] = df['Cluster'] == -1

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X.toarray())
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]

    return df

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Log Analyzer", layout="wide")
st.title("AI-Powered Log Anomaly Detector")

uploaded_file = st.file_uploader("Upload your .log or .csv file", type=["log", "csv"])
mode = st.radio("Choose processing mode:", ["Fast", "Full"])

if uploaded_file is not None:
    filename = uploaded_file.name
    st.success(f"File uploaded: {filename}")

    # Read file
    if filename.endswith(".log"):
        content = StringIO(uploaded_file.getvalue().decode("utf-8")).readlines()
        df = pd.DataFrame({'Log': [line.strip() for line in content if line.strip()]})
    else:
        df = pd.read_csv(uploaded_file)
        if 'Log' not in df.columns:
            df.columns = ['Log']

    df['Cleaned'] = df['Log'].apply(clean_log)

    st.info("Clustering logs... this may take a few moments.")
    with st.spinner("Running DBSCAN clustering..."):
        clustered_df = cluster_logs(df.copy(), fast_mode=(mode == "Fast"))

    # Visualization
    st.subheader("Anomaly Visualization (DBSCAN Clusters)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=clustered_df,
        x='pca1', y='pca2',
        hue='Is_Anomaly',
        palette={True: 'red', False: 'blue'},
        style='Is_Anomaly',
        s=80
    )
    plt.title("PCA of Log Clusters (Anomalies in Red)")
    st.pyplot(fig)

    # Show data
    st.subheader("Anomalous Logs (Cluster = -1)")
    st.dataframe(clustered_df[clustered_df['Is_Anomaly']][['Log', 'Cluster']].head(50))

    # Download
    csv = clustered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Logs CSV",
        data=csv,
        file_name="clustered_logs.csv",
        mime='text/csv'
    )
