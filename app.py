import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import random
import io
import requests

np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="InCURA", page_icon="data/Logo_incura.svg", layout="wide")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("data/Logo_incura.svg", width=120)
with col2:
    st.title("Integrative Gene Clustering based on Transcription Factor Binding Sites")
    
    st.markdown(
    "InCURA enables clustering of differentially expressed genes (DEGs) based on shared transcription factor binding patterns. "
    "Paste a list of DEGs and either all expressed genes or a list of transcription factors of interest to explore regulatory modules, "
    "visualize gene clusters, and identify enriched TF binding sites.\n\n"
    "**Note:** This implementation of InCURA uses a pre-computed TF binding site matrix "
    "with a fixed background model based on all protein coding genes in the respective organism. For more versatile functionality use the [GitHub version of InCURA](https://github.com/saezlab/incura)."
    )

# -------------------------------
# Dataset selection
# -------------------------------
dataset_choice = st.radio(
    "Select organism:",
    options=["Mouse", "Human"],
    index=0,
    horizontal=True
)

# -------------------------------
# Load Preprocessed TFBS Matrix
# -------------------------------
@st.cache_data
def load_preprocessed_matrix(species: str) -> pd.DataFrame:
    urls = {
        "Mouse": "https://zenodo.org/records/15866266/files/full_count_matrix_mouse.tsv?download=1",
        "Human": "https://zenodo.org/records/15866266/files/full_count_matrix_human.tsv?download=1"
    }
    url = urls.get(species)
    if not url:
        raise ValueError("Invalid species selection")

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download matrix. Status: {response.status_code}")

    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")), sep="\t", index_col=0)
    df.index = df.index.str.lower()  # lowercase gene names
    df.columns = df.columns.str.lower()  # lowercase TF names
    return df

with st.spinner("Loading TFBS matrix..."):
    matrix = load_preprocessed_matrix(dataset_choice)

# --- Gene Input ---
st.subheader("Filtering for Differentially Expressed Genes")
rows_text = st.text_area("Paste gene names here (one gene per line):", placeholder="Gene1\nGene2\nGene3")
row_list_raw = [x.strip() for x in rows_text.replace(',', '\n').splitlines() if x.strip()]
row_list = [x.lower() for x in row_list_raw]
valid_rows = [r for r in row_list if r in matrix.index]
st.markdown(f"**Genes pasted:** {len(row_list_raw)} &nbsp;&nbsp;|&nbsp;&nbsp; **Valid genes:** {len(valid_rows)}")

# --- TF Input ---
st.subheader("Filtering for Transcription Factors")
cols_text = st.text_area("Paste TF names here (or all expressed genes, one per line):", placeholder="TF1\nTF2\nTF3")
col_list_raw = [x.strip() for x in cols_text.replace(',', '\n').splitlines() if x.strip()]
col_list = [x.lower() for x in col_list_raw]
valid_cols = [c for c in col_list if c in matrix.columns]
st.markdown(f"**TFs pasted:** {len(col_list_raw)} &nbsp;&nbsp;|&nbsp;&nbsp; **Valid TFs:** {len(valid_cols)}")

# ---------------------------------
# TF enrichment 
# ---------------------------------
@st.cache_data
def tfbs_cluster_enrichment(binary_matrix, cluster_labels, pval_threshold=0.05):
    unique_clusters = np.unique(cluster_labels)
    enrichment_results = []

    for cluster_label in unique_clusters:
        cluster_genes = binary_matrix[cluster_labels == cluster_label]
        background_genes = binary_matrix[cluster_labels != cluster_label]

        cluster_size = len(cluster_genes)
        background_size = len(background_genes)

        for tfbs in binary_matrix.columns:
            cluster_tfbs_count = cluster_genes[tfbs].sum()
            background_tfbs_count = background_genes[tfbs].sum()

            contingency_table = [
                [cluster_tfbs_count, cluster_size - cluster_tfbs_count],
                [background_tfbs_count, background_size - background_tfbs_count],
            ]
            _, p_value = fisher_exact(contingency_table)

            enrichment_results.append({
                'TFBS': tfbs,
                'Cluster': cluster_label,
                'p_value': p_value
            })

    enrichment_df = pd.DataFrame(enrichment_results)
    enrichment_df['corrected_pval'] = multipletests(enrichment_df['p_value'], method='fdr_bh')[1]
    significant_results = enrichment_df[enrichment_df['corrected_pval'] < pval_threshold]

    return significant_results

# --- Filter + Show ---
if valid_rows and valid_cols:
    count_matrix = matrix.loc[valid_rows, valid_cols]
    count_matrix = count_matrix.astype(np.float32)
    st.success(f"Filtered to matrix with shape: {count_matrix.shape}")

    if count_matrix.empty:
        st.warning("No matching rows found after filtering. Check gene and TF names.")
    else:
        # --- UMAP ---
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(count_matrix.values)

        # --- KMeans ---
        n_clusters = st.slider("Number of clusters (KMeans)", min_value=2, max_value=10, value=4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(count_matrix.values)

        # --- Plot UMAP with Clusters ---
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', s=50)
        ax.set_title("UMAP Projection with KMeans Clusters")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        st.pyplot(fig)

        # --- Show cluster assignments ---
        clustered_df = pd.DataFrame({
            "gene": count_matrix.index,
            "cluster": cluster_labels
        })

        # Capitalise gene/TF names if desired
        if dataset_choice == "Mouse":
            clustered_df["gene"] = clustered_df["gene"].str.capitalize()
        elif dataset_choice == "Human":
            clustered_df["gene"] = clustered_df["gene"].str.upper()

        st.subheader("Cluster Assignments")
        st.dataframe(clustered_df)

        # --- Download ---
        csv = clustered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cluster Assignments (CSV)",
            data=csv,
            file_name="incura_gene_clusters.csv",
            mime="text/csv"
        )

        # --- Optional: KMeans performance metrics ---
        st.subheader("Optional: Evaluate  Clustering Performance")
        run_metrics = st.checkbox("Run KMeans metrics (Inertia & Silhouette) to determine optimal k")

        if run_metrics:
            st.write("Calculating performance metrics across k=2 to k=10...")
            X = count_matrix.values
            k_values = range(2, 10)
            inertias = []
            silhouette_scores = []

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)
                score = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
                silhouette_scores.append(score)

            fig, ax1 = plt.subplots(figsize=(8, 5))
            color = 'tab:blue'
            ax1.set_xlabel('Number of clusters (k)', fontsize=14)
            ax1.set_ylabel('Inertia', color=color, fontsize=14)
            ax1.plot(k_values, inertias, marker='o', color=color, label='Inertia')
            ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
            ax1.tick_params(axis='x', labelsize=12)

            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Silhouette Score', color=color, fontsize=14)
            ax2.plot(k_values, silhouette_scores, marker='s', linestyle='--', color=color, label='Silhouette Score')
            ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

            plt.title('K-Means Performance Metrics', fontsize=16)
            fig.tight_layout()
            st.pyplot(fig)

        # --- Optional: TFBS Enrichment ---
        st.subheader("Identify Enriched TFBS Driving the Clustering")
        run_enrichment = st.checkbox("Run TFBS enrichment analysis")

        if run_enrichment:
            st.write("Running TFBS enrichment analysis per cluster...")

            binary_matrix = count_matrix.astype(bool).astype(int)  # ensure binary
            cluster_series = pd.Series(cluster_labels, index=count_matrix.index)

            enrichment_df = tfbs_cluster_enrichment(binary_matrix, cluster_series)

            if enrichment_df.empty:
                st.warning("No significantly enriched TFBS found at the specified threshold.")
            else:
                tfbs_counts = enrichment_df.groupby("TFBS")["Cluster"].nunique()
                ubiquitous_tfbs = tfbs_counts[tfbs_counts > 1].index
                enrichment_df = enrichment_df[~enrichment_df["TFBS"].isin(ubiquitous_tfbs)]

                top_tfbs = (
                    enrichment_df.groupby("Cluster")
                    .apply(lambda x: x.nsmallest(10, "p_value"))
                    .reset_index(drop=True)
                )
                enrichment_df = enrichment_df[enrichment_df["TFBS"].isin(top_tfbs["TFBS"])]
                
                # Format TF names in final enrichment_df before plotting
                if dataset_choice == "Mouse":
                    enrichment_df["TFBS_display"] = enrichment_df["TFBS"].apply(lambda x: x.capitalize())
                elif dataset_choice == "Human":
                    enrichment_df["TFBS_display"] = enrichment_df["TFBS"].apply(lambda x: x.upper())
                else:
                    enrichment_df["TFBS_display"] = enrichment_df["TFBS"]

                # Pivot and plot
                pivot_df = enrichment_df.pivot(index="TFBS_display", columns="Cluster", values="corrected_pval")
                pivot_df = pivot_df.sort_values(by=pivot_df.columns.tolist())

                fig, ax = plt.subplots(figsize=(5, max(4, 0.3 * len(pivot_df))))
                sns.heatmap(-np.log10(pivot_df), cmap="viridis", annot=False, ax=ax)
                ax.set_title("-log10(corrected p-values) of TFBS enrichment")
                ax.set_xlabel("Cluster", fontsize=14)
                ax.set_ylabel("TFBS", fontsize=14)
                st.pyplot(fig)

                csv_enrich = enrichment_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Enrichment Results (CSV)",
                    data=csv_enrich,
                    file_name="incura_tfbs_enrichment.csv",
                    mime="text/csv"
                )
else:
    st.warning("Please paste valid gene and TF names.")

st.markdown(
    """
    <hr style="margin-top: 2em; margin-bottom: 1em">
    <div style='text-align: center; font-size: 0.85em; color: gray;'>
        © 2025 InCURA. By Lorna Wessels. Developed for academic research purposes.
    </div>
    """,
    unsafe_allow_html=True
)
