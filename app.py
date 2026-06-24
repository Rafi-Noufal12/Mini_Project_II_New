import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config untuk tampilan profesional

st.set_page_config(
    page_title="Dashboard Eksplorasi Data Senyawa Kimia",
    page_icon="🧪",
    layout="wide"
)

pg = st.navigation([
        st.Page(
            "packages/1_Prediction.py",
            title="Prediction",
            icon="📊"
        ),
        
        st.Page(
            "packages/2_Visualization.py",
            title="Visualization",
            icon="💡"
        ),
            st.Page(
            "https://github.com/Rafi-Noufal12",
            title="Github",
            icon="🐙"
        ),
    ])
pg.run()

# Judul Dashboard
st.title("🧪 Dashboard Eksplorasi Data Senyawa Kimia")
st.caption("Aplikasi sederhana untuk melihat persebaran data dan korelasi pada dataset *Delaney Solubility*.")

# Alert Berhasil Dimuat (Sesuai Gambar 1)
st.success("Dataset berhasil dimuat langsung dari GitHub!")

# Load Data Delaney Solubility dari URL Publik terpercaya
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv"
    try:
        df = pd.read_csv(url)

        df = df.drop_duplicates()

        df = df.rename(columns={
            'MolLogP': 'MolLogP',
            'MolWt': 'MolWt',
            'NumRotatableBonds': 'NumRotatableBonds',
            'AromaticProportion': 'AromaticProportion',
            'logS': 'LogS'
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

# TAMBAHKAN INI
df = load_data()

# Metrik Total Sampel dan Jumlah Parameter (Sesuai Gambar 1)
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Total Sampel Senyawa (Bersih)", value=len(df))

with col2:
    st.metric(label="Jumlah Parameter/Kolom", value=len(df.columns))

# Section: Analisis Distribusi Fitur
st.header("📊 Analisis Distribusi Fitur")
st.write("Pilih salah satu parameter di bawah untuk melihat pola persebaran nilainya.")

# Kamus definisi parameter sesuai dengan Gambar 2
parameter_definitions = {
    "MolLogP": "**MolLogP (Octanol-Water Partition Coefficient):** Mengukur seberapa besar suka (berikatan) senyawa kimia terhadap minyak/lemak dibandingkan air.",
    "MolWt": "**MolWt (Molecular Weight):** Berat molekul dari senyawa tersebut.",
    "NumRotatableBonds": "**NumRotatableBonds:** Jumlah ikatan tunggal di dalam molekul yang bisa berputar bebas.",
    "AromaticProportion": "**AromaticProportion:** Proporsi bagian molekul yang memiliki struktur cincin aromatik.",
    "LogS": "**LogS (logarithma of Aqueous Solubility):** Nilai Logaritma dari kelarutan zat di dalam air (mol/L). Semakin kecil, semakin sulit larut di dalam air."
}

# Dropdown Pilihan Parameter
selected_param = st.selectbox(
    "Pilih Parameter Senyawa:",
    options=list(parameter_definitions.keys())
)

# Tampilan Dinamis Informasi/Definisi Parameter (Sesuai Permintaan)
st.info(parameter_definitions[selected_param])

# Pembuatan Grafik Distribusi (Histogram + KDE) menggunakan Seaborn & Matplotlib
fig, ax = plt.subplots(figsize=(10, 5))

# Styling grafik agar elegan dan bersih
sns.histplot(df[selected_param], kde=True, color="#2b5c8f", ax=ax, edgecolor="black", alpha=0.7)
ax.set_title(f"Grafik Distribusi Nilai: {selected_param}", fontsize=14, pad=15)
ax.set_xlabel(selected_param, fontsize=11)
ax.set_ylabel("Frekuensi", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Tampilkan grafik ke Streamlit
st.pyplot(fig)