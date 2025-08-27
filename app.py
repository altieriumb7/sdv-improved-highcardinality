import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# our simple constraints (post-generation)
# option 1: import from the package root (after step 2)
from new_sdv_2 import RangeConstraint, UniqueConstraint
from new_sdv_2.constraints import apply_constraints

st.set_page_config(page_title="new_sdv • Gaussian Copula", layout="wide")

st.title("new_sdv • Gaussian Copula Synthesizer")
st.markdown("Carica un CSV, configura **più vincoli** e genera dati sintetici.")

# --- Sidebar: upload & settings
st.sidebar.header("Dati & Impostazioni")
uploaded = st.sidebar.file_uploader("Carica un CSV", type=["csv"])
use_example = st.sidebar.checkbox("Usa dataset di esempio", value=False)

random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)
seed = int(random_state)
import random, numpy as np
random.seed(seed)
np.random.seed(seed)
try:
    import torch
    torch.manual_seed(seed)
except Exception:
    pass

n_rows = st.sidebar.number_input("Righe sintetiche", min_value=1, value=500, step=10)

st.sidebar.header("Aiuto vincoli")
if st.sidebar.checkbox("Mostra guida", value=False):
    st.sidebar.info(
        "• RangeConstraint: impone min/max su una colonna numerica (clampa i valori fuori range).\n"
        "• UniqueConstraint: impone valori unici per una o più colonne."
    )

# --- Load data
df = None
if use_example:
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "income": rng.gamma(shape=2.0, scale=25000, size=600),
        "age": rng.randint(18, 80, size=600),
        "score": rng.beta(a=2.0, b=5.0, size=600),
        "city": rng.choice(["Roma", "Milano", "Napoli", "Torino"], p=[0.5, 0.3, 0.15, 0.05], size=600)
    })
elif uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Errore nel leggere il CSV: {e}")

if df is None:
    st.info("Carica un CSV oppure seleziona 'Usa dataset di esempio' nella sidebar.")
    st.stop()

# SDV (>=1.4) non supporta dtype 'category' in input
cat_dtype_cols = df.select_dtypes(include=["category"]).columns
if len(cat_dtype_cols):
    df[cat_dtype_cols] = df[cat_dtype_cols].astype(object)

st.subheader("Anteprima dati reali")
st.dataframe(df.head(), use_container_width=True)

# Quick stats
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Righe", len(df))
with c2:
    st.metric("Colonne", df.shape[1])
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
with c3:
    st.metric("Numeriche", int(len(num_cols)))
with c4:
    st.metric("Categoriche", int(len(cat_cols)))

# EDA
with st.expander("Esplora distribuzioni (reali)"):
    if num_cols:
        sel_num = st.multiselect("Colonne numeriche", num_cols, default=num_cols[: min(3, len(num_cols))], key="eda_num")
        for i, c in enumerate(sel_num):
            st.plotly_chart(
                px.histogram(df, x=c, nbins=30, title=f"Reale • {c}"),
                use_container_width=True,
                key=f"real_hist_{c}_{i}"
            )
    if cat_cols:
        sel_cat = st.multiselect("Colonne categoriche", cat_cols, default=cat_cols[: min(3, len(cat_cols))], key="eda_cat")
        for j, c in enumerate(sel_cat):
            counts = df[c].value_counts(dropna=False).reset_index()
            counts.columns = [c, "conteggio"]
            st.plotly_chart(
                px.bar(counts, x=c, y="conteggio", title=f"Reale • {c}"),
                use_container_width=True,
                key=f"real_bar_{c}_{j}"
            )

# ==========================
#    MULTI-CONSTRAINT UI
# ==========================
st.subheader("Configura vincoli (multipli)")

constraints = []

num_cols_all = num_cols
all_cols = df.columns.tolist()

with st.expander("RangeConstraint (multipli)"):
    st.caption("Seleziona **una o più** colonne numeriche; per ciascuna imposta Min/Max.")
    sel_range_cols = st.multiselect("Colonne numeriche da vincolare", num_cols_all, key="range_cols_select")

    if sel_range_cols:
        for idx, col in enumerate(sel_range_cols):
            col_min = float(np.nanmin(df[col].values))
            col_max = float(np.nanmax(df[col].values))
            m1, m2 = st.columns(2)
            with m1:
                vmin = st.number_input(f"[{col}] Min", value=col_min, key=f"rc_min_{col}_{idx}")
            with m2:
                vmax = st.number_input(f"[{col}] Max", value=col_max, key=f"rc_max_{col}_{idx}")
            if vmin <= vmax:
                constraints.append(RangeConstraint(col, min_val=vmin, max_val=vmax))
            else:
                st.warning(f"[{col}] Min > Max: correggi i limiti.")

with st.expander("UniqueConstraint (multipli)"):
    st.caption("Seleziona **una o più** colonne da rendere uniche (deduplica le righe in base a quella colonna).")
    sel_unique_cols = st.multiselect("Colonne univoche", all_cols, key="uniq_cols_select")
    for uc in sel_unique_cols:
        constraints.append(UniqueConstraint(uc))

# ==========================
#       GENERAZIONE
# ==========================
st.subheader("Generazione sintetica")
if st.button("Genera dati sintetici", key="btn_generate"):
    try:
        # Build metadata (SDV >= 1.4 requires it)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        synth = GaussianCopulaSynthesizer(metadata)
        synth.fit(df)
        synth_df = synth.sample(int(n_rows))

        # Apply our simple constraints post-hoc
        synth_df = apply_constraints(synth_df, constraints)

        st.success(f"Generati {len(synth_df)} record sintetici")
        t1, t2, t3 = st.tabs(["Anteprima", "Confronto correlazioni", "Download"])

        with t1:
            st.dataframe(synth_df.head(), use_container_width=True, height=250)
            # confronto veloce per prima num/cat
            if num_cols:
                c = num_cols[0]
                st.plotly_chart(px.histogram(df, x=c, nbins=30, title=f"Reale • {c}"), use_container_width=True, key=f"cmp_real_hist_{c}")
                st.plotly_chart(px.histogram(synth_df, x=c, nbins=30, title=f"Sintetico • {c}"), use_container_width=True, key=f"cmp_synth_hist_{c}")
            if cat_cols:
                c = cat_cols[0]
                r_counts = df[c].value_counts(dropna=False).reset_index()
                r_counts.columns = [c, "conteggio"]
                s_counts = synth_df[c].value_counts(dropna=False).reset_index()
                s_counts.columns = [c, "conteggio"]
                st.plotly_chart(px.bar(r_counts, x=c, y="conteggio", title=f"Reale • {c}"), use_container_width=True, key=f"cmp_real_bar_{c}")
                st.plotly_chart(px.bar(s_counts, x=c, y="conteggio", title=f"Sintetico • {c}"), use_container_width=True, key=f"cmp_synth_bar_{c}")

        with t2:
            rnum = df.select_dtypes(include=[np.number])
            snum = synth_df.select_dtypes(include=[np.number])
            if rnum.shape[1] >= 2:
                st.plotly_chart(px.imshow(rnum.corr(), text_auto=True, title="Correlazione • Reale"), use_container_width=True, key="corr_real")
            if snum.shape[1] >= 2:
                st.plotly_chart(px.imshow(snum.corr(), text_auto=True, title="Correlazione • Sintetico"), use_container_width=True, key="corr_synth")

        with t3:
            st.download_button(
                "Scarica CSV sintetico",
                data=synth_df.to_csv(index=False).encode("utf-8"),
                file_name="synthetic.csv",
                mime="text/csv",
                key="download_csv",
            )
    except Exception as e:
        st.error(f"Errore durante la generazione: {e}")
        st.exception(e)
