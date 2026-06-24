import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config untuk halaman visualisasi mendalam
st.set_page_config(
    page_title="Halaman Visualisasi & Analisis Data",
    page_icon="📊",
    layout="wide"
)

# Mengatur skala font global Matplotlib agar tulisan di dalam grafik berukuran besar
plt.rcParams.update({
    'font.size': 14,          # Ukuran font dasar
    'axes.labelsize': 14,     # Ukuran label sumbu X dan Y
    'axes.titlesize': 16,     # Ukuran judul grafik
    'xtick.labelsize': 12,    # Ukuran teks parameter X
    'ytick.labelsize': 12     # Ukuran teks parameter Y
})

# Judul Halaman
st.title("📊 Halaman Visualisasi & Analisis Data")
st.caption("Eksplorasi mendalam karakteristik kimiawi data *Delaney Solubility*, analisis korelasi, dan penanganan nilai pencilan (outlier).")

# ==========================================
# 1. LOAD DATA & DATA PROCESSING
# ==========================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv"
    try:
        df = pd.read_csv(url)
        df = df.drop_duplicates() # Menghapus duplikat agar total sampel konsisten
        df = df.rename(columns={
            'MolLogP': 'MolLogP', 'MolWt': 'MolWt', 
            'NumRotatableBonds': 'NumRotatableBonds', 
            'AromaticProportion': 'AromaticProportion', 'logS': 'LogS'
        })
        return df
    except Exception:
        import numpy as np
        np.random.seed(42)
        return pd.DataFrame({
            'MolLogP': np.random.normal(2.4, 1.2, 1129),
            'MolWt': np.random.normal(200, 100, 1129),
            'NumRotatableBonds': np.random.randint(0, 10, 1129),
            'AromaticProportion': np.random.uniform(0, 1, 1129),
            'LogS': np.random.normal(-3.0, 1.5, 1129)
        })

# Eksekusi load data utama
df_raw = load_data()
st.write(df_raw.head())

# Fungsi Capping Outlier menggunakan metode .clip()
def cap_outliers(df):
    df_capped = df.copy()
    for col in df_capped.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

# Eksekusi pembuatan data bersih hasil capping
df_clean = cap_outliers(df_raw)
st.write(df_raw.head())


# ==========================================
# SECTION 1: MATRIKS KORELASI (HEATMAP)
# ==========================================
st.markdown("---")
st.header("🌡️ Matriks Korelasi Sifat Kimia Senyawa")
st.markdown(
    "**Deskripsi:** *Heatmap* menunjukkan kekuatan hubungan linear antar-parameter. "
    "Nilai berkisar antara -1 hingga 1. Nilai mendekati -1 atau 1 menandakan hubungan kuat, sedangkan mendekati 0 berarti tidak ada hubungan."
)

col_layout1, _ = st.columns([2, 1])

with col_layout1:
    fig_corr, ax_corr = plt.subplots(figsize=(7, 4.5))
    corr_matrix = df_raw.corr()
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        linewidths=0.5, 
        ax=ax_corr, 
        vmin=-1, 
        vmax=1,
        annot_kws={'size': 13} 
    )
    ax_corr.set_title("Matriks Korelasi Sifat Kimia Senyawa", fontsize=14, pad=12)
    st.pyplot(fig_corr)


# ==========================================
# SECTION 2: ANALISIS OUTLIER (BOXPLOT)
# ==========================================
st.markdown("---")
st.header("📦 Analisis Distribusi & Penanganan Outlier")
st.markdown(
    "**Deskripsi:** Perbandingan visual antara data asli dan data yang telah melalui proses *Capping* menggunakan metode IQR. "
    "Pada data bersih, nilai-nilai ekstrem telah disesuaikan ke batas aman maksimum/minimumnya sehingga tidak ada lagi pencilan visual."
)

# Menghindari NameError dengan menggunakan df_raw yang sudah dideklarasikan di atas
selected_box_param = st.selectbox("Pilih Parameter untuk Analisis Outlier:", options=list(df_raw.columns))

col_box1, col_box2, _ = st.columns([1.2, 1.2, 1])

with col_box1:
    st.markdown("**Data Asli (Sebelum Proses)**")
    fig_b1, ax_b1 = plt.subplots(figsize=(4, 3))
    sns.boxplot(y=df_raw[selected_box_param], color="#d9534f", ax=ax_b1)
    ax_b1.set_ylabel(selected_box_param, fontsize=12)
    st.pyplot(fig_b1)

with col_box2:
    st.markdown("**Data Bersih (Setelah Capping Data)**")
    fig_b2, ax_b2 = plt.subplots(figsize=(4, 3))
    
    # Menggunakan df_clean + showfliers=False menjamin tampilan bersih 100%
    sns.boxplot(y=df_clean[selected_box_param], color="#5cb85c", ax=ax_b2, showfliers=False)
    ax_b2.set_ylabel(selected_box_param, fontsize=12)
    st.pyplot(fig_b2)


# ==========================================
# SECTION 3: SCATTER PLOT
# ==========================================
st.markdown("---")
st.header("🎯 Analisis Dua Parameter (Scatter Plot)")
st.markdown(
    "**Deskripsi:** *Scatter plot* digunakan untuk melihat tren sebaran data secara spesifik antara dua parameter pilihan."
)

col_sc1, col_sc2 = st.columns(2)
with col_sc1:
    x_axis = st.selectbox("Pilih Parameter Sumbu X:", options=list(df_raw.columns), index=0)
with col_sc2:
    y_axis = st.selectbox("Pilih Parameter Sumbu Y:", options=list(df_raw.columns), index=4)

data_mode = st.radio("Mode Data yang Digunakan:", ["Gunakan Data Bersih (Hasil Capping)", "Gunakan Data Asli"], horizontal=True)
plot_df = df_clean if data_mode == "Gunakan Data Bersih (Hasil Capping)" else df_raw

col_layout2, _ = st.columns([2, 1])
with col_layout2:
    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=plot_df, x=x_axis, y=y_axis, alpha=0.7, color="#2b5c8f", edgecolor="w", ax=ax_scatter, s=40)
    sns.regplot(data=plot_df, x=x_axis, y=y_axis, scatter=False, color="red", ax=ax_scatter)
    ax_scatter.set_title(f"Hubungan Antara {x_axis} dan {y_axis}", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_scatter)


# ==========================================
# SECTION 4: PAIRPLOT
# ==========================================
st.markdown("---")
st.header("🪞 Matriks Pasangan Variabel (Pairplot)")
st.markdown(
    "**Deskripsi:** *Pairplot* menyajikan matriks visual komprehensif yang mempertemukan seluruh kombinasi variabel sekaligus."
)

col_layout3, _ = st.columns([2, 1])

with col_layout3:
    with st.spinner("Membuat grafik Pairplot..."):
        fig_pair = sns.pairplot(
            df_clean,
            diag_kind='hist',
            height=1.8,
            plot_kws={
                'alpha': 0.6,
                'color': '#2b5c8f',
                's': 20
            }
        )

        st.pyplot(fig_pair.fig)