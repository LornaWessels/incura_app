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
    st.markdown("""
    InCURA enables clustering of differentially expressed genes (DEGs) based on shared transcription factor binding patterns. 
    Paste a list of DEGs and either all expressed genes or a list of transcription factors of interest to explore regulatory modules, 
    visualize gene clusters, and identify enriched TF binding sites.

    **Note:** This implementation of InCURA uses a pre-computed TF binding site matrix 
    with a fixed background model based on all protein coding genes in the respective organism. For more versatile functionality use the [GitHub version of InCURA](https://github.com/saezlab/incura).
    """)

# -------------------------------
# Dataset selection
# -------------------------------
dataset_choice = st.radio(
    "Select organism:",
    options=["Mouse", "Human"],
    index=0,
    horizontal=True
)

@st.cache_resource
def load_df(species: str):
    urls = {
        "Mouse": "https://zenodo.org/records/15862228/files/fimo_mouse.parquet?download=1",
        "Human": "https://zenodo.org/records/15862228/files/fimo_human.parquet?download=1"
    }
    url = urls.get(species)
    if not url:
        raise ValueError("Invalid species selection")

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file from Zenodo. Status code: {response.status_code}")

    df = pd.read_parquet(io.BytesIO(response.content))
    df["gene_lower"] = df["gene"].str.lower()
    df["motif_id_lower"] = df["motif_id"].str.lower()
    return df

# Load the dataset based on user selection
with st.spinner("Loading TFBS matrix. May take up to 2 minutes..."):
    df = load_df(dataset_choice)

# --- Row Input ---
st.subheader("Filtering for Differentially Expressed Genes")
rows_text = st.text_area("Paste gene names here (one gene per line):", placeholder="Gene1\nGene2\nGene3")
row_list = [x.strip().lower() for x in rows_text.replace(',', '\n').splitlines() if x.strip()]

# --- Column Input ---
st.subheader("Filtering for Transcription Factors")
cols_text = st.text_area("Paste TF names here (or list of all expressed genes, one per line):", placeholder="TF1\nTF2\nTF3")
col_list = [x.strip().lower() for x in cols_text.replace(',', '\n').splitlines() if x.strip()]

# --- Filter original df --- #
if row_list and col_list:
    df = df[df['motif_id_lower'].isin(col_list) & df['gene_lower'].isin(row_list)]

    if df.empty:
        st.warning("No matching rows found after filtering. Check gene and TF names.")
    else:
        df = df.sort_values(by=['gene', 'motif_id', 'strand', 'start'])

        # Collapse overlapping motifs per gene/motif/strand
        summarized = []
        prev = None
        for row in df.itertuples(index=False):
            gene, motif, strand, start, stop, score = row.gene, row.motif_id, row.strand, row.start, row.stop, row.score
            if prev and (gene, motif, strand) == prev[:3] and start <= prev[4]:
                prev[4] = max(prev[4], stop)
                prev[5] += score
                prev[6] += 1
            else:
                if prev:
                    summarized.append(prev)
                prev = [gene, motif, strand, start, stop, score, 1]
        if prev:
            summarized.append(prev)

        del df
        summary_df = pd.DataFrame(summarized, columns=['symbols', 'motif', 'strand', 'start', 'end', 'score_sum', 'count'])
        summary_df['score'] = summary_df['score_sum'] / summary_df['count']
        summary_df = summary_df[['symbols', 'motif', 'strand', 'start', 'end', 'score']]

        st.write(f"Summarized to {len(summary_df)} regions.")

        # Count TF occurrences
        matrix = summary_df.groupby(['symbols', 'motif']).size().unstack(fill_value=0).astype(np.int16)
        matrix = matrix.loc[:, sorted(matrix.columns)]
        del summary_df

        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(matrix.values)

        n_clusters = st.slider("Number of clusters (KMeans)", 2, 10, 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix.values)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=50)
        ax.set(title="UMAP Projection with KMeans Clusters", xlabel="UMAP 1", ylabel="UMAP 2")
        st.pyplot(fig)

        clustered_df = pd.DataFrame({"gene": matrix.index, "cluster": labels})
        st.subheader("Cluster Assignments")
        st.dataframe(clustered_df)

        csv = clustered_df.to_csv(index=False).encode("utf-8")
        st.download_button("\ud83d\udc45 Download Cluster Assignments (CSV)", csv, "incura_gene_clusters.csv", "text/csv")

        if st.checkbox("Run TFBS enrichment analysis"):
            binary_matrix = (matrix > 0).astype(np.int8)
            enrichment_results = []

            for cluster in np.unique(labels):
                cluster_genes = binary_matrix[labels == cluster]
                bg_genes = binary_matrix[labels != cluster]
                for tf in binary_matrix.columns:
                    fg_pos = cluster_genes[tf].sum()
                    fg_neg = len(cluster_genes) - fg_pos
                    bg_pos = bg_genes[tf].sum()
                    bg_neg = len(bg_genes) - bg_pos
                    table = [[fg_pos, fg_neg], [bg_pos, bg_neg]]
                    _, p = fisher_exact(table)
                    enrichment_results.append({"TFBS": tf, "Cluster": cluster, "p_value": p})

            enrich_df = pd.DataFrame(enrichment_results)
            enrich_df['corrected_pval'] = multipletests(enrich_df['p_value'], method='fdr_bh')[1]
            enrich_df = enrich_df[enrich_df['corrected_pval'] < 0.05]

            if enrich_df.empty:
                st.warning("No significantly enriched TFBS found.")
            else:
                tfbs_counts = enrich_df.groupby("TFBS")['Cluster'].nunique()
                unique_tfbs = tfbs_counts[tfbs_counts == 1].index
                enrich_df = enrich_df[enrich_df['TFBS'].isin(unique_tfbs)]
                top_tfbs = enrich_df.groupby("Cluster").apply(lambda x: x.nsmallest(10, 'p_value')).reset_index(drop=True)
                heatmap_df = top_tfbs.pivot(index='TFBS', columns='Cluster', values='corrected_pval')

                fig, ax = plt.subplots(figsize=(5, max(4, 0.3 * len(heatmap_df))))
                sns.heatmap(-np.log10(heatmap_df), cmap="viridis", ax=ax)
                ax.set(title="-log10(corrected p-values) of TFBS enrichment", xlabel="Cluster", ylabel="TFBS")
                st.pyplot(fig)

                csv_enrich = enrich_df.to_csv(index=False).encode("utf-8")
                st.download_button("\ud83d\udc45 Download Enrichment Results (CSV)", csv_enrich, "incura_tfbs_enrichment.csv", "text/csv")
else:
    st.warning("Please paste valid gene and TF names.")

st.markdown("""
    <hr style="margin-top: 2em; margin-bottom: 1em">
    <div style='text-align: center; font-size: 0.85em; color: gray;'>
        © 2025 InCURA. By Lorna Wessels. Developed for academic research purposes.
    </div>
    """, unsafe_allow_html=True)