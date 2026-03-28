"""
Dashboard Interattiva – PMI Europee (Eurostat SBS)
Capitolo 4 – Tesi LM31 | Università Mercatorum

Avvio:  doppio clic su avvia_dashboard.bat

Versione 2.0 – Metodologia corretta:
  * Split temporale (no split casuale)
  * Cross-validation con TimeSeriesSplit (5 fold)
  * Baseline model (mediana storica per paese-settore)
  * Clustering con z-score relativo per settore
  * Feature importance salvata e visualizzata
"""

import os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PERCORSI
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "Data")
os.makedirs(MODEL_DIR, exist_ok=True)

INDICATORS = ["V11110", "V12110", "V12120", "V12130", "V12150"]
IND_LABELS = {
    "V11110": "Numero di imprese",
    "V12110": "Fatturato (M€)",
    "V12120": "Valore della produzione (M€)",
    "V12130": "Valore aggiunto (M€)",
    "V12150": "Spese per il personale (M€)",
}
# Breve descrizione di ciascun indicatore (mostrata come nota nella sidebar)
IND_DESCRIPTIONS = {
    "V11110": "Quante PMI sono attive per settore e paese in ciascun anno",
    "V12110": "Ricavi totali da vendita di beni e servizi (in milioni di euro)",
    "V12120": "Fatturato + variazione scorte + lavori in economia (M€)",
    "V12130": "Ricchezza generata: produzione meno costi intermedi (M€)",
    "V12150": "Salari, stipendi e oneri sociali pagati ai dipendenti (M€)",
}

# ─────────────────────────────────────────────────────────────────────────────
# DIZIONARIO CODICI NACE Rev. 2  (Eurostat SBS)
# Fonte: Eurostat / NACE Rev. 2 – Regolamento CE n. 1893/2006
# ─────────────────────────────────────────────────────────────────────────────
NACE_LABELS = {
    # ── Sezioni standard ──────────────────────────────────────────────────────
    "A":          "Agricoltura, silvicoltura e pesca",
    "B":          "Estrazione di minerali da cave e miniere",
    "C":          "Attività manifatturiere",
    "D":          "Fornitura di energia elettrica, gas e vapore",
    "E":          "Fornitura di acqua; reti fognarie, rifiuti",
    "F":          "Costruzioni",
    "G":          "Commercio all'ingrosso e al dettaglio",
    "H":          "Trasporto e magazzinaggio",
    "I":          "Alloggio e ristorazione",
    "J":          "Servizi di informazione e comunicazione (ICT)",
    "K":          "Attività finanziarie e assicurative",
    "L":          "Attività immobiliari",
    "M":          "Attività professionali, scientifiche e tecniche",
    "N":          "Noleggio, agenzie di viaggio e servizi di supporto",
    "O":          "Amministrazione pubblica e difesa",
    "P":          "Istruzione",
    "Q":          "Sanità e assistenza sociale",
    "R":          "Attività artistiche, sportive e di intrattenimento",
    "S":          "Altre attività di servizi",
    "T":          "Attività di famiglie come datori di lavoro",
    "U":          "Organizzazioni e organismi extraterritoriali",
    # ── Aggregati Eurostat SBS ────────────────────────────────────────────────
    "B-E":            "Industria (escluse costruzioni): B+C+D+E",
    "B-F":            "Industria e costruzioni: B+C+D+E+F",
    "B-N":            "Economia delle imprese (B–N)",
    "B-N_S95_X_K":    "Economia delle imprese – tutti i settori escluso finanza (K)",
    "B-N_X_K":        "Economia delle imprese escluso finanza (K)",
    "C10-C12":        "Manifattura: alimentari, bevande e tabacco",
    "C13-C15":        "Manifattura: tessile, abbigliamento e pelle",
    "C16-C18":        "Manifattura: legno, carta e stampa",
    "C19-C23":        "Manifattura: chimica, farmaceutica, gomma e plastica",
    "C24-C25":        "Manifattura: metallurgia e prodotti in metallo",
    "C26-C28":        "Manifattura: elettronica, macchinari e apparecchiature",
    "C29-C30":        "Manifattura: autoveicoli e altri mezzi di trasporto",
    "C31-C33":        "Manifattura: mobili, altre industrie e riparazione",
    "G-J":            "Commercio, trasporti, alloggio e ICT (G+H+I+J)",
    "G-N":            "Servizi alle imprese (G–N)",
    "G45":            "Commercio e riparazione di autoveicoli",
    "G46":            "Commercio all'ingrosso (escluso autoveicoli)",
    "G47":            "Commercio al dettaglio (escluso autoveicoli)",
    "HT":             "Manifattura ad alta tecnologia (High-Tech)",
    "MHT":            "Manifattura a media-alta tecnologia",
    "MLT":            "Manifattura a media-bassa tecnologia",
    "LT":             "Manifattura a bassa tecnologia",
    "KIA":            "Attività knowledge-intensive (KIA) – tutti i settori",
    "KIABI":          "Servizi ad alta intensità di conoscenza – industria",
    "KIABI_X_K":      "KIA – industria escluso finanza (K)",
    "KIABI_X_K_R90":  "KIA – industria escluso finanza (K) e arti/spettacolo (R90)",
    "LKIA":           "Attività a bassa intensità di conoscenza",
    "SBI":            "Servizi a supporto delle imprese",
    "S95":            "Riparazione di computer e beni personali",
    "M_N":            "Servizi professionali e amministrativi (M+N)",
    "J58-J60":        "Editoria, audiovisivo e radiodiffusione",
    "J61":            "Telecomunicazioni",
    "J62-J63":        "Informatica e servizi di informazione",
    "M69-M70":        "Attività legali, contabili e consulenza direzionale",
    "M71":            "Architettura, ingegneria e collaudi tecnici",
    "M72":            "Ricerca e sviluppo (R&S)",
    "M73":            "Pubblicità e ricerche di mercato",
    "M74-M75":        "Altre attività professionali e veterinarie",
    "N77":            "Attività di noleggio e leasing",
    "N78":            "Agenzie di selezione e somministrazione di personale",
    "N79":            "Agenzie di viaggio e tour operator",
    "N80-N82":        "Vigilanza, pulizia e servizi di supporto",
    "R90-R92":        "Arti, intrattenimento e attività ricreative",
    "R93":            "Sport, divertimento e attività ricreative",
    "S94":            "Attività di organizzazioni associative",
    "S96":            "Altre attività di servizi alla persona",
}

def nace_label(code: str) -> str:
    """Restituisce 'CODICE – Descrizione' oppure solo 'CODICE' se non trovato."""
    desc = NACE_LABELS.get(code, "")
    return f"{code} – {desc}" if desc else code
RISK_COLORS = {
    "Bassa Performance": "#e74c3c",
    "Media Performance": "#f39c12",
    "Alta Performance":  "#27ae60",
}
RANDOM_STATE  = 50
CAT_FEATURES  = ["nace_r2", "geo"]
NUM_FEATURES  = ["TIME_PERIOD"]
TARGET        = "OBS_VALUE"
N_CV_SPLITS   = 5


# ─────────────────────────────────────────────────────────────────────────────
# GESTIONE MODELLI
# ─────────────────────────────────────────────────────────────────────────────
def _pulisci_modelli():
    import glob
    for f in glob.glob(os.path.join(MODEL_DIR, "*.pkl")) + \
             glob.glob(os.path.join(MODEL_DIR, "*.csv")):
        try:
            os.remove(f)
        except Exception:
            pass


def _modelli_validi():
    import joblib
    required = (
        [os.path.join(MODEL_DIR, f"best_model_{i}.pkl") for i in INDICATORS]
        + [os.path.join(MODEL_DIR, f"kmeans_{i}.pkl") for i in INDICATORS]
        + [os.path.join(MODEL_DIR, "all_results.pkl"),
           os.path.join(MODEL_DIR, "confronto_modelli.csv"),
           os.path.join(MODEL_DIR, "feature_importance.pkl")]
    )
    if not all(os.path.exists(p) for p in required):
        return False
    try:
        joblib.load(os.path.join(MODEL_DIR, f"best_model_{INDICATORS[0]}.pkl"))
        return True
    except Exception:
        _pulisci_modelli()
        return False


_modelli_presenti = _modelli_validi


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-TRAINING v2
# ─────────────────────────────────────────────────────────────────────────────
def _addestra_tutti(bar, status):
    import joblib
    from sklearn.pipeline        import Pipeline
    from sklearn.preprocessing   import OneHotEncoder, StandardScaler
    from sklearn.compose         import ColumnTransformer
    from sklearn.linear_model    import Ridge
    from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network  import MLPRegressor
    from sklearn.cluster         import KMeans
    from sklearn.metrics         import (mean_absolute_error, mean_squared_error,
                                         r2_score, silhouette_score)
    from sklearn.model_selection import TimeSeriesSplit

    # 1 – Carica dataset completo
    status.text("Caricamento dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "dataset_sbs_clean.csv"))

    # 2 – Split temporale: 80% anni per train, 20% piu' recenti per test
    anni_s      = sorted(df["TIME_PERIOD"].unique())
    n_train     = max(1, int(len(anni_s) * 0.8))
    ANNO_CUT    = anni_s[n_train - 1]
    status.text(f"Split: train <= {ANNO_CUT} | test > {ANNO_CUT}")

    splits = {}
    for ind in INDICATORS:
        di = df[df["indic_sb"] == ind].copy()
        mtr = di["TIME_PERIOD"] <= ANNO_CUT
        mte = di["TIME_PERIOD"] >  ANNO_CUT
        ft  = CAT_FEATURES + NUM_FEATURES
        splits[ind] = dict(
            X_tr=di[mtr][ft].reset_index(drop=True),
            X_te=di[mte][ft].reset_index(drop=True),
            y_tr=di[mtr][TARGET].reset_index(drop=True),
            y_te=di[mte][TARGET].reset_index(drop=True),
        )

    def mk_pre():
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ])

    def met(yt, yp):
        return dict(
            MAE=float(mean_absolute_error(yt, yp)),
            RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
            R2=float(r2_score(yt, yp)),
        )

    # 3 – Baseline: mediana storica per (geo, nace_r2)
    status.text("Calcolo baseline...")
    bl_met = {}
    for ind in INDICATORS:
        s   = splits[ind]
        ref = (pd.concat([s["X_tr"], s["y_tr"].rename(TARGET)], axis=1)
               .groupby(["geo", "nace_r2"])[TARGET].median())
        ybl = (s["X_te"].join(ref, on=["geo", "nace_r2"])[TARGET]
               .fillna(float(s["y_tr"].median())))
        bl_met[ind] = met(s["y_te"], ybl)

    # 4 – Modelli ML
    cfg = {
        "Ridge":             Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=200, max_depth=10,
                                 min_samples_split=5,
                                 random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=200, learning_rate=0.05,
                                 max_depth=4, subsample=0.8,
                                 random_state=RANDOM_STATE),
        "MLP Regressor":     MLPRegressor(
                                 hidden_layer_sizes=(128, 64, 32),
                                 activation="relu", solver="adam",
                                 max_iter=1000, early_stopping=True,
                                 random_state=RANDOM_STATE),
    }

    tscv    = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    N_tot   = len(cfg) * len(INDICATORS) + len(INDICATORS)
    step    = 0
    res_te, res_cv, pipes = {}, {}, {}

    # 5 – Training + TimeSeriesSplit CV
    for nome, obj in cfg.items():
        res_te[nome], res_cv[nome], pipes[nome] = {}, {}, {}
        for ind in INDICATORS:
            status.text(f"{nome} -> {ind}...")
            s  = splits[ind]
            # ordina per anno prima di splittare
            ord_ = s["X_tr"]["TIME_PERIOD"].argsort().values
            Xs   = s["X_tr"].iloc[ord_].reset_index(drop=True)
            ys   = s["y_tr"].iloc[ord_].reset_index(drop=True)

            cv_r2, cv_mae, cv_rmse = [], [], []
            for itr, ival in tscv.split(Xs):
                pc = Pipeline([("pre", mk_pre()),
                               ("mod", obj.__class__(**obj.get_params()))])
                pc.fit(Xs.iloc[itr], ys.iloc[itr])
                mc = met(ys.iloc[ival], pc.predict(Xs.iloc[ival]))
                cv_r2.append(mc["R2"])
                cv_mae.append(mc["MAE"])
                cv_rmse.append(mc["RMSE"])

            res_cv[nome][ind] = dict(
                R2_mean=float(np.mean(cv_r2)),   R2_std=float(np.std(cv_r2)),
                MAE_mean=float(np.mean(cv_mae)),  MAE_std=float(np.std(cv_mae)),
                RMSE_mean=float(np.mean(cv_rmse)),RMSE_std=float(np.std(cv_rmse)),
            )

            pf = Pipeline([("pre", mk_pre()),
                           ("mod", obj.__class__(**obj.get_params()))])
            pf.fit(s["X_tr"], s["y_tr"])
            res_te[nome][ind] = met(s["y_te"], pf.predict(s["X_te"]))
            pipes[nome][ind]  = pf

            step += 1
            bar.progress(step / N_tot)

    # 6 – Feature importance
    status.text("Feature importance...")
    fi_data = {}
    for nome in ["Random Forest", "Gradient Boosting"]:
        fi_ind = {}
        for ind in INDICATORS:
            pip      = pipes[nome][ind]
            ohe      = pip.named_steps["pre"].named_transformers_["cat"]
            fout     = list(ohe.get_feature_names_out(CAT_FEATURES)) + NUM_FEATURES
            imps     = pip.named_steps["mod"].feature_importances_
            n_nace   = sum(1 for f in fout if f.startswith("nace_r2_"))
            n_geo    = sum(1 for f in fout if f.startswith("geo_"))
            fi_ind[ind] = {
                "Settore (nace_r2)":  float(imps[:n_nace].sum()),
                "Paese (geo)":        float(imps[n_nace:n_nace + n_geo].sum()),
                "Anno (TIME_PERIOD)": float(imps[-1]),
            }
        fi_data[nome] = fi_ind
    joblib.dump(fi_data, os.path.join(MODEL_DIR, "feature_importance.pkl"))

    # 7 – Salva miglior modello per indicatore
    status.text("Salvataggio modelli...")
    for ind in INDICATORS:
        best = max(cfg.keys(), key=lambda n: res_te[n][ind]["R2"])
        pb   = Pipeline([("pre", mk_pre()),
                         ("mod", cfg[best].__class__(**cfg[best].get_params()))])
        pb.fit(splits[ind]["X_tr"], splits[ind]["y_tr"])
        joblib.dump(dict(
            model=pb,
            name=best,
            metriche=res_te[best][ind],
            cv_metriche=res_cv[best][ind],
            baseline_metriche=bl_met[ind],
            anno_cutoff=int(ANNO_CUT),
        ), os.path.join(MODEL_DIR, f"best_model_{ind}.pkl"))

    joblib.dump(dict(
        **res_te,
        cv=res_cv, baseline=bl_met,
        IND_LABELS=IND_LABELS, anno_cutoff=int(ANNO_CUT),
    ), os.path.join(MODEL_DIR, "all_results.pkl"))

    righe = []
    for ind in INDICATORS:
        righe.append(dict(
            Indicatore=ind, Descrizione=IND_LABELS[ind],
            Modello="Baseline (Mediana Storica)",
            MAE=round(bl_met[ind]["MAE"], 2),
            RMSE=round(bl_met[ind]["RMSE"], 2),
            R2=round(bl_met[ind]["R2"], 4),
            R2_CV_media="", R2_CV_std="",
        ))
        for nome in cfg:
            righe.append(dict(
                Indicatore=ind, Descrizione=IND_LABELS[ind],
                Modello=nome,
                MAE=round(res_te[nome][ind]["MAE"], 2),
                RMSE=round(res_te[nome][ind]["RMSE"], 2),
                R2=round(res_te[nome][ind]["R2"], 4),
                R2_CV_media=round(res_cv[nome][ind]["R2_mean"], 4),
                R2_CV_std=round(res_cv[nome][ind]["R2_std"], 4),
            ))
    pd.DataFrame(righe).to_csv(
        os.path.join(MODEL_DIR, "confronto_modelli.csv"), index=False)

    # 8 – Clustering con z-score per settore
    for ind in INDICATORS:
        status.text(f"K-Means -> {ind}...")
        df_i = df[df["indic_sb"] == ind].copy()

        # Statistiche calcolate SOLO sul training (no leakage)
        stats = (df_i[df_i["TIME_PERIOD"] <= ANNO_CUT]
                 .groupby("nace_r2")[TARGET]
                 .agg(["mean", "std"])
                 .rename(columns={"mean": "mu", "std": "sigma"}))
        stats["sigma"] = stats["sigma"].replace(0, 1).fillna(1)

        df_i = df_i.join(stats, on="nace_r2")
        df_i["mu"]    = df_i["mu"].fillna(df_i[TARGET].median())
        df_i["sigma"] = df_i["sigma"].fillna(1).replace(0, 1)
        df_i["zsc"]   = ((df_i[TARGET] - df_i["mu"])
                         / df_i["sigma"]).clip(-3, 3)

        sc   = StandardScaler()
        Xcl  = sc.fit_transform(df_i[["zsc", "TIME_PERIOD"]])
        km   = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
        lb   = km.fit_predict(Xcl)

        sil  = float(silhouette_score(Xcl, lb))
        rank = km.cluster_centers_[:, 0].argsort()
        lmap = {
            int(rank[0]): "Bassa Performance",
            int(rank[1]): "Media Performance",
            int(rank[2]): "Alta Performance",
        }
        joblib.dump(dict(km=km, scaler=sc, sector_stats=stats,
                         label_map=lmap, silhouette=sil),
                    os.path.join(MODEL_DIR, f"kmeans_{ind}.pkl"))

        step += 1
        bar.progress(step / N_tot)

    status.text("Completato!")


# ─────────────────────────────────────────────────────────────────────────────
# PAGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PMI Dashboard · Eurostat", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ── Badge di rischio ─────────────────────────────── */
.risk-bassa{background:#e74c3c;color:white;padding:5px 12px;
            border-radius:6px;font-weight:700;}
.risk-media{background:#f39c12;color:white;padding:5px 12px;
            border-radius:6px;font-weight:700;}
.risk-alta {background:#27ae60;color:white;padding:5px 12px;
            border-radius:6px;font-weight:700;}
.mb{background:#f8f9fa;border-left:4px solid #2980b9;padding:8px 12px;
    border-radius:4px;margin:4px 0;font-size:0.88em;line-height:1.6;}

/* ── Barra intestazione ───────────────────────────── */
.header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, #002060 0%, #003580 60%, #004aad 100%);
    padding: 10px 24px;
    border-radius: 8px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.header-left {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.header-student {
    color: #ffffff;
    font-size: 1.05em;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.header-matricola {
    color: #a8c8ff;
    font-size: 0.82em;
    font-weight: 400;
    letter-spacing: 0.2px;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 14px;
}
.header-right img {
    height: 46px;
    width: auto;
    filter: brightness(1.08);
    border-radius: 4px;
}
.header-relatore {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 2px;
}
.header-rel-label {
    color: #a8c8ff;
    font-size: 0.76em;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.header-rel-name {
    color: #ffffff;
    font-size: 0.95em;
    font-weight: 700;
    white-space: nowrap;
}
/* Rimuove il padding default di Streamlit per non lasciare spazio bianco sopra */
[data-testid="stAppViewContainer"] > .main > div:first-child {
    padding-top: 0.5rem;
}
</style>""", unsafe_allow_html=True)

# ── Carica logo come base64 (embedding locale, nessuna dipendenza esterna) ────
import base64 as _b64
_LOGO_B64 = ""
try:
    _logo_path = os.path.join(BASE_DIR, "Unimercatorum_logo.svg.png")
    with open(_logo_path, "rb") as _lf:
        _LOGO_B64 = _b64.b64encode(_lf.read()).decode()
except Exception:
    pass  # se il file non c'e', la barra mostra solo il testo

# ── Barra intestazione ────────────────────────────────────────────────────────
_logo_tag = (f'<img src="data:image/png;base64,{_LOGO_B64}" alt="Unimercatorum">'
             if _LOGO_B64 else "")

st.markdown(f"""
<div class="header-bar">
  <div class="header-left">
    <span class="header-student">Marco Ottoboni</span>
    <span class="header-matricola">Matricola: 0312400775</span>
  </div>
  <div class="header-right">
    <div class="header-relatore">
      <span class="header-rel-label">Relatore</span>
      <span class="header-rel-name">Prof. Mario Fabio Polidoro</span>
    </div>
    {_logo_tag}
  </div>
</div>
""", unsafe_allow_html=True)

# ── PRIMO AVVIO ───────────────────────────────────────────────────────────────
if not _modelli_presenti():
    st.title("PMI Dashboard – Eurostat SBS")
    st.info("Prima esecuzione: addestramento modelli in corso (2-3 minuti).")
    st.markdown("""
    La dashboard eseguira' automaticamente:
    - **Split temporale** (80% train / 20% test per anno, nessuno split casuale)
    - **Cross-validation** con 5 fold temporali (TimeSeriesSplit)
    - **Baseline model** (mediana storica per paese-settore)
    - **4 modelli ML**: Ridge, Random Forest, Gradient Boosting, MLP
    - **Clustering** con normalizzazione z-score per settore
    """)
    bar = st.progress(0)
    msg = st.empty()
    try:
        _addestra_tutti(bar, msg)
        st.success("Modelli pronti! Ricarico...")
        st.rerun()
    except Exception as e:
        st.error(f"Errore: {e}")
        st.info("Verifica: pip install -r requirements.txt | "
                "Data/processed/dataset_sbs_clean.csv presente?")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# CARICAMENTO (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def carica_dati():
    return pd.read_csv(
        os.path.join(DATA_DIR, "processed", "dataset_sbs_clean.csv"))

@st.cache_resource
def carica_modello(ind):
    import joblib
    p = os.path.join(MODEL_DIR, f"best_model_{ind}.pkl")
    if not os.path.exists(p):
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None

@st.cache_resource
def carica_kmeans(ind):
    import joblib
    p = os.path.join(MODEL_DIR, f"kmeans_{ind}.pkl")
    if not os.path.exists(p):
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None

@st.cache_data
def carica_risultati():
    import joblib
    p = os.path.join(MODEL_DIR, "all_results.pkl")
    if not os.path.exists(p):
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None

@st.cache_data
def carica_confronto():
    p = os.path.join(MODEL_DIR, "confronto_modelli.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def carica_fi():
    import joblib
    p = os.path.join(MODEL_DIR, "feature_importance.pkl")
    if not os.path.exists(p):
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


df_full = carica_dati()
paesi   = sorted(df_full["geo"].unique())
settori = sorted(df_full["nace_r2"].unique())
anni    = sorted(df_full["TIME_PERIOD"].unique())

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## PMI Dashboard\n**Eurostat SBS**")
    st.markdown("---")

    # ── Selettore indicatore con descrizione breve ────────────────────────────
    ind_sel = st.selectbox(
        "📌 Indicatore",
        INDICATORS,
        format_func=lambda x: f"{x} – {IND_LABELS[x]}",
    )
    # Nota esplicativa sull'indicatore selezionato
    st.caption(f"ℹ️ {IND_DESCRIPTIONS[ind_sel]}")

    # ── Selettore paese ───────────────────────────────────────────────────────
    paese = st.selectbox(
        "🌍 Paese",
        paesi,
        index=paesi.index("IT") if "IT" in paesi else 0,
    )

    # ── Selettore settore NACE con descrizione nella tendina ──────────────────
    st.markdown(
        "**🏭 Settore NACE** "
        "<span style='font-size:0.78em;color:#888;'>"
        "(classificazione europea delle attività economiche)</span>",
        unsafe_allow_html=True,
    )
    settore = st.selectbox(
        "Settore NACE",
        settori,
        format_func=nace_label,
        label_visibility="collapsed",   # nasconde etichetta duplicata
    )
    # Mostra la descrizione estesa del settore selezionato
    if settore in NACE_LABELS:
        st.caption(f"ℹ️ {NACE_LABELS[settore]}")

    st.markdown("---")
    st.markdown(
        f"**{len(df_full):,}** osservazioni\n\n"
        f"**{df_full['geo'].nunique()}** paesi | "
        f"**{df_full['nace_r2'].nunique()}** settori\n\n"
        f"Anni: {int(min(anni))}–{int(max(anni))}"
    )
    pay0 = carica_modello(INDICATORS[0])
    if pay0 and "anno_cutoff" in pay0:
        ac = pay0["anno_cutoff"]
        st.caption(f"Train ≤{ac} | Test >{ac} | CV {N_CV_SPLITS} fold")
    st.caption("Tesi LM31 · Università Mercatorum")

# ── HEADER + KPI ──────────────────────────────────────────────────────────────
st.markdown("# Dashboard Interattiva – PMI Europee (Eurostat SBS)")
_nace_desc = NACE_LABELS.get(settore, "")
st.markdown(
    f"**Indicatore:** `{ind_sel}` – *{IND_LABELS[ind_sel]}*  |  "
    f"**Paese:** `{paese}`  |  **Settore:** `{settore}`"
    + (f" – *{_nace_desc}*" if _nace_desc else "")
)
st.caption(f"📊 {IND_DESCRIPTIONS[ind_sel]}")
st.markdown("---")

df_ind = df_full[df_full["indic_sb"] == ind_sel]
df_fil = df_ind[(df_ind["geo"] == paese) & (df_ind["nace_r2"] == settore)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Osservazioni filtrate", len(df_fil))
c2.metric("Media storica",
          f"{df_fil['OBS_VALUE'].mean():,.0f}" if len(df_fil) else "N/D")
c3.metric("Paesi nel dataset",   df_ind["geo"].nunique())
c4.metric("Settori nel dataset", df_ind["nace_r2"].nunique())
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Previsione ML", "Early Warning", "Confronto Modelli",
    "Esplorazione Dataset", "🏢 Valutazione Aziendale"])


# ══ TAB 1 – PREVISIONE ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("### Previsione con il modello ML")
    pay = carica_modello(ind_sel)
    if pay is None:
        st.warning("Modello non trovato. Riavvia la dashboard.")
    else:
        col1, col2 = st.columns([1, 1.8])
        with col1:
            st.markdown(f"**Modello attivo:** `{pay['name']}`")
            m = pay["metriche"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE",     f"{m['MAE']:,.2f}")
            mc2.metric("RMSE",    f"{m['RMSE']:,.2f}")
            mc3.metric("R² test", f"{m['R2']:.4f}")

            if "cv_metriche" in pay:
                cv = pay["cv_metriche"]
                st.markdown(
                    f"<div class='mb'>CV {N_CV_SPLITS} fold temporali<br>"
                    f"R² = {cv['R2_mean']:.4f} ± {cv['R2_std']:.4f}<br>"
                    f"RMSE = {cv['RMSE_mean']:,.2f} ± {cv['RMSE_std']:,.2f}</div>",
                    unsafe_allow_html=True)

            if "baseline_metriche" in pay:
                bl = pay["baseline_metriche"]
                st.markdown(
                    f"<div class='mb'>Baseline (mediana storica)<br>"
                    f"R² = {bl['R2']:.4f} | RMSE = {bl['RMSE']:,.2f}</div>",
                    unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Inserisci i parametri")
            p_in = st.selectbox("Paese", paesi,
                                index=paesi.index(paese), key="p1")
            s_in = st.selectbox("Settore NACE", settori,
                                index=settori.index(settore),
                                format_func=nace_label, key="s1")
            a_in = st.number_input("Anno", min_value=2000, max_value=2035,
                                   value=int(max(anni)), key="a1")

            if st.button("Genera previsione", type="primary", use_container_width=True):
                Xn = pd.DataFrame({"nace_r2": [s_in], "geo": [p_in],
                                   "TIME_PERIOD": [a_in]})
                try:
                    y_hat = float(pay["model"].predict(Xn)[0])
                    st.markdown("---")
                    st.success(f"**Valore previsto: {y_hat:,.2f}**")
                    st.caption(f"{IND_LABELS[ind_sel]} | {p_in} | {s_in} | {a_in}")

                    kd = carica_kmeans(ind_sel)
                    if kd:
                        try:
                            if "sector_stats" in kd and s_in in kd["sector_stats"].index:
                                mu    = float(kd["sector_stats"].loc[s_in, "mu"])
                                sigma = float(kd["sector_stats"].loc[s_in, "sigma"])
                            else:
                                mu, sigma = float(y_hat), 1.0
                            sigma  = max(sigma, 1e-6)
                            zscore = float(np.clip((y_hat - mu) / sigma, -3, 3))
                            Xcl    = kd["scaler"].transform([[zscore, a_in]])
                            prof   = kd["label_map"][int(kd["km"].predict(Xcl)[0])]
                            css    = {"Bassa Performance": "risk-bassa",
                                      "Media Performance": "risk-media",
                                      "Alta Performance":  "risk-alta"}[prof]
                            st.markdown("**Profilo vs media del settore:**")
                            st.markdown(f'<span class="{css}">{prof}</span>',
                                        unsafe_allow_html=True)
                            sil = kd.get("silhouette", None)
                            st.caption(
                                f"Z-score settore {s_in}: {zscore:+.2f}σ | "
                                f"Media settore: {mu:,.2f}"
                                + (f" | Silhouette: {sil:.3f}" if sil else ""))
                        except Exception:
                            pass
                except Exception as ex:
                    st.error(f"Errore: {ex}")

        with col2:
            st.markdown("#### Serie storica")
            if len(df_fil):
                fig = px.line(df_fil.sort_values("TIME_PERIOD"),
                              x="TIME_PERIOD", y="OBS_VALUE", markers=True,
                              color_discrete_sequence=["#2980b9"],
                              labels={"TIME_PERIOD": "Anno",
                                      "OBS_VALUE": IND_LABELS[ind_sel]})
                fig.update_layout(height=310, margin=dict(l=0, r=0, t=20, b=0),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nessun dato storico per la selezione.")

            st.markdown("#### Top paesi – ultimo anno")
            ult  = int(df_ind["TIME_PERIOD"].max())
            dm   = (df_ind[df_ind["TIME_PERIOD"] == ult]
                    .groupby("geo")["OBS_VALUE"].mean().reset_index())
            fig2 = px.bar(dm.sort_values("OBS_VALUE", ascending=False).head(20),
                          x="geo", y="OBS_VALUE", color="OBS_VALUE",
                          color_continuous_scale="Blues",
                          labels={"geo": "Paese", "OBS_VALUE": IND_LABELS[ind_sel]})
            fig2.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0),
                               coloraxis_showscale=False,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)


# ══ TAB 2 – EARLY WARNING ════════════════════════════════════════════════════
with tab2:
    st.markdown("### Early Warning – Profili di Performance")
    kd = carica_kmeans(ind_sel)
    if kd is None:
        st.warning("Modello K-Means non trovato. Riavvia la dashboard.")
    else:
        sil = kd.get("silhouette", None)
        ki1, ki2, ki3 = st.columns(3)
        ki1.metric("Algoritmo", "K-Means  (k=3)")
        ki2.metric("Silhouette Score",
                   f"{sil:.3f}" if sil is not None else "N/D",
                   help="Valori > 0.3 indicano cluster ben separati")
        ki3.metric("Normalizzazione", "Z-score per settore")

        if sil is not None:
            q = ("Buona separazione" if sil > 0.4
                 else "Separazione accettabile" if sil > 0.2
                 else "Cluster sovrapposti")
            st.caption(f"Qualita' clustering: {q} (silhouette = {sil:.3f})")

        st.info(
            "Il profilo di performance e' calcolato tramite z-score rispetto "
            "alla media storica del settore (nace_r2). Una PMI manifatturiera "
            "viene confrontata con la media del manifatturiero, non con settori "
            "diversi.")

        # Clustering con z-score
        df_cl = df_ind.copy()
        if "sector_stats" in kd:
            df_cl = df_cl.join(kd["sector_stats"], on="nace_r2")
            df_cl["mu"]    = df_cl["mu"].fillna(df_cl["OBS_VALUE"].median())
            df_cl["sigma"] = df_cl["sigma"].fillna(1).replace(0, 1)
            df_cl["zsc"]   = ((df_cl["OBS_VALUE"] - df_cl["mu"])
                              / df_cl["sigma"]).clip(-3, 3)
            Xcl = kd["scaler"].transform(df_cl[["zsc", "TIME_PERIOD"]])
        else:
            Xcl = kd["scaler"].transform(df_cl[["OBS_VALUE", "TIME_PERIOD"]])

        df_cl["cluster"] = kd["km"].predict(Xcl)
        df_cl["profilo"] = df_cl["cluster"].map(kd["label_map"])

        c1, c2 = st.columns([1, 1.6])
        with c1:
            dist = df_cl["profilo"].value_counts().reset_index()
            dist.columns = ["Profilo", "N"]
            dist["Percentuale"] = (dist["N"] / len(df_cl) * 100).round(1)
            fig_p = px.pie(dist, names="Profilo", values="N",
                           color="Profilo",
                           color_discrete_map=RISK_COLORS, hole=0.4)
            fig_p.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_p, use_container_width=True)
            st.dataframe(dist.style.format({"Percentuale": "{:.1f}%"}),
                         use_container_width=True, hide_index=True)
        with c2:
            fig_sc = px.scatter(
                df_cl.sort_values("profilo"),
                x="TIME_PERIOD", y="OBS_VALUE", color="profilo",
                color_discrete_map=RISK_COLORS,
                hover_data=["geo", "nace_r2"], opacity=0.6,
                labels={"TIME_PERIOD": "Anno",
                        "OBS_VALUE":   IND_LABELS[ind_sel],
                        "profilo":     "Profilo"})
            fig_sc.update_traces(marker_size=6)
            fig_sc.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0),
                                 legend=dict(orientation="h",
                                             yanchor="bottom", y=1.02),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("---")
        prof_s = st.radio("Filtra per profilo:",
                          ["Bassa Performance", "Media Performance",
                           "Alta Performance"],
                          horizontal=True)
        dr = (df_cl[df_cl["profilo"] == prof_s]
              .groupby("geo")["OBS_VALUE"]
              .agg(["mean", "count"]).reset_index()
              .rename(columns={"geo": "Paese", "mean": "Media", "count": "N obs"}))
        dr["Media"] = dr["Media"].round(2)
        dr = dr.sort_values("Media", ascending=(prof_s == "Bassa Performance"))
        if len(dr):
            fig_r = px.bar(dr.head(25), x="Paese", y="Media",
                           color_discrete_sequence=[RISK_COLORS[prof_s]],
                           labels={"Media": IND_LABELS[ind_sel]})
            fig_r.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_r, use_container_width=True)
            st.dataframe(dr, use_container_width=True, hide_index=True)


# ══ TAB 3 – CONFRONTO MODELLI ════════════════════════════════════════════════
with tab3:
    st.markdown("### Confronto performance modelli ML")
    df_conf = carica_confronto()
    all_res = carica_risultati()
    fi_data = carica_fi()

    if df_conf is None or all_res is None:
        st.warning("File risultati non trovati. Riavvia la dashboard.")
    else:
        st.info(
            "Metriche calcolate su **test set temporale** (anni piu' recenti) "
            "e tramite **5-fold TimeSeriesSplit** sul training set. "
            "Il Baseline e' la mediana storica per paese-settore: "
            "i modelli ML devono superarlo per essere utili.")

        df_ci = df_conf[df_conf["Indicatore"] == ind_sel].copy()

        c1, c2 = st.columns([1.5, 1.5])
        with c1:
            st.markdown(f"#### Metriche – `{ind_sel}`")
            show_cols = [c for c in
                         ["Modello", "MAE", "RMSE", "R2",
                          "R2_CV_media", "R2_CV_std"]
                         if c in df_ci.columns]
            st.dataframe(
                df_ci[show_cols]
                .style
                .highlight_max(subset=[c for c in ["R2", "R2_CV_media"]
                                        if c in show_cols], color="#d4edda")
                .highlight_min(subset=[c for c in ["MAE", "RMSE"]
                                        if c in show_cols], color="#d4edda")
                .format({c: ("{:,.2f}" if c in ("MAE", "RMSE") else
                             ("{:.4f}"  if c == "R2" else "{}"))
                         for c in show_cols if c != "Modello"}),
                use_container_width=True, hide_index=True)

        with c2:
            if "R2" in df_ci.columns:
                fig_r2 = px.bar(
                    df_ci, x="Modello", y="R2", color="Modello",
                    text="R2",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title=f"R² su test set – {ind_sel}")
                fig_r2.update_traces(texttemplate="%{text:.4f}",
                                     textposition="outside")
                fig_r2.update_layout(
                    height=320, margin=dict(l=0, r=0, t=40, b=60),
                    showlegend=False, xaxis_tickangle=-20,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("---")
        st.markdown("#### R² su tutti gli indicatori")
        mn  = [k for k in all_res
               if k not in ("IND_LABELS", "cv", "baseline", "anno_cutoff")]
        r2d = [{"Modello": m, "Indicatore": i,
                "R² Test": all_res[m][i].get("R2", 0)}
               for m in mn for i in INDICATORS]
        if "baseline" in all_res:
            r2d += [{"Modello": "Baseline", "Indicatore": i,
                     "R² Test": all_res["baseline"][i].get("R2", 0)}
                    for i in INDICATORS]
        fig_all = px.bar(
            pd.DataFrame(r2d), x="Indicatore", y="R² Test",
            color="Modello", barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_all.update_layout(
            height=340, margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_all, use_container_width=True)

        # Feature Importance
        if fi_data:
            st.markdown("---")
            st.markdown("#### Feature Importance")
            st.markdown(
                "Contributo relativo di ogni variabile alle previsioni. "
                "Settore e Paese sono codificati con One-Hot Encoding: "
                "la barra mostra la somma dell'importanza di tutte le categorie.")
            fc1, fc2 = st.columns(2)
            for col_fi, nm_fi in zip([fc1, fc2],
                                     ["Random Forest", "Gradient Boosting"]):
                with col_fi:
                    st.markdown(f"**{nm_fi}** – `{ind_sel}`")
                    if nm_fi in fi_data and ind_sel in fi_data[nm_fi]:
                        fi_i  = fi_data[nm_fi][ind_sel]
                        df_fi = (pd.DataFrame.from_dict(
                                     fi_i, orient="index",
                                     columns=["Importanza"])
                                 .reset_index()
                                 .rename(columns={"index": "Feature"}))
                        df_fi = df_fi.sort_values("Importanza", ascending=True)
                        fig_fi = px.bar(
                            df_fi, x="Importanza", y="Feature",
                            orientation="h",
                            text=df_fi["Importanza"].map("{:.1%}".format),
                            color="Feature",
                            color_discrete_sequence=[
                                "#27ae60", "#e74c3c", "#3498db"])
                        fig_fi.update_traces(textposition="outside")
                        fig_fi.update_layout(
                            height=220, margin=dict(l=0, r=70, t=10, b=0),
                            showlegend=False,
                            xaxis_tickformat=".0%",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_fi, use_container_width=True)


# ══ TAB 4 – ESPLORAZIONE ══════════════════════════════════════════════════════
with tab4:
    st.markdown("### Esplorazione Dataset Eurostat SBS")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Mappa geografica")
        anno_m = st.selectbox("Anno", sorted(anni, reverse=True), key="ma")
        dm = (df_full[(df_full["indic_sb"] == ind_sel) &
                      (df_full["TIME_PERIOD"] == anno_m)]
              .groupby("geo")["OBS_VALUE"].mean().reset_index())
        _iso = {
            "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE",
            "DE": "DEU", "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP",
            "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN", "IE": "IRL",
            "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT",
            "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
            "SI": "SVN", "SK": "SVK", "NO": "NOR", "IS": "ISL", "CH": "CHE",
            "TR": "TUR", "MK": "MKD", "RS": "SRB", "ME": "MNE", "AL": "ALB",
            "BA": "BIH", "XK": "XKX", "UK": "GBR", "GB": "GBR",
        }
        dm["geo_iso3"] = dm["geo"].map(_iso)
        dm_map = dm.dropna(subset=["geo_iso3"])
        fig_map = px.choropleth(
            dm_map, locations="geo_iso3", locationmode="ISO-3",
            color="OBS_VALUE", color_continuous_scale="Blues", scope="europe",
            labels={"OBS_VALUE": IND_LABELS[ind_sel], "geo_iso3": "Paese"},
            title=f"{ind_sel} – {anno_m}")
        fig_map.update_layout(height=370, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    with c2:
        st.markdown("#### Trend per settore")
        top8 = (df_ind.groupby("nace_r2")["OBS_VALUE"].mean()
                .sort_values(ascending=False).head(8).index.tolist())
        ss = st.multiselect("Settori", top8, default=top8[:4], key="ts")
        if ss:
            dt = (df_ind[df_ind["nace_r2"].isin(ss)]
                  .groupby(["TIME_PERIOD", "nace_r2"])["OBS_VALUE"]
                  .mean().reset_index())
            fig_tr = px.line(
                dt, x="TIME_PERIOD", y="OBS_VALUE",
                color="nace_r2", markers=True,
                labels={"TIME_PERIOD": "Anno",
                        "OBS_VALUE": IND_LABELS[ind_sel],
                        "nace_r2": "Settore"})
            fig_tr.update_layout(
                height=350, margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.info("Seleziona almeno un settore.")

    st.markdown("---")
    st.markdown("#### Dati filtrati")
    df_raw = (df_full[(df_full["indic_sb"] == ind_sel) & (df_full["geo"] == paese)]
              .sort_values("TIME_PERIOD", ascending=False))
    st.dataframe(df_raw.reset_index(drop=True).head(50), use_container_width=True)
    st.caption(f"Max 50 righe su {len(df_raw)} per {paese} – {ind_sel}")


# ══ TAB 5 – VALUTAZIONE AZIENDALE ════════════════════════════════════════════
with tab5:
    import plotly.graph_objects as go

    st.markdown("### 🏢 Valutazione Aziendale")
    st.markdown(
        "Inserisci i dati della tua azienda per confrontarli con il **benchmark "
        "ML** del settore di riferimento. Il sistema calcola lo scostamento, "
        "il z-score rispetto alla media storica e classifica la performance.")

    # ── Nota metodologica ──────────────────────────────────────────────────────
    with st.expander("ℹ️ Come funziona questa analisi", expanded=False):
        st.markdown("""
**Benchmark ML** – Il modello di regressione (Ridge / Random Forest / Gradient Boosting / MLP)
addestrato su dati Eurostat prevede il valore atteso per un'azienda con le caratteristiche
specificate (paese, settore, anno). Questo valore rappresenta il *benchmark di settore*.

**Gap** – Differenza assoluta e percentuale tra il valore reale inserito e il benchmark previsto.

**Z-score** – Scostamento dalla media storica del settore, espresso in deviazioni standard
(σ). Valori > +1σ indicano performance sopra la media; < -1σ sotto la media.

**Profilo di performance** – Classificazione K-Means (k=3) sui dati normalizzati:
- 🟢 **Alta Performance** – cluster a z-score elevato
- 🟡 **Media Performance** – cluster intermedio
- 🔴 **Bassa Performance** – cluster a z-score basso

> *I parametri del clustering sono calcolati esclusivamente sul training set (anni ≤ anno_cutoff),
> garantendo assenza di data leakage.*
""")

    st.markdown("---")

    # ── Form di input ──────────────────────────────────────────────────────────
    st.markdown("#### Dati azienda da valutare")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        val_paese = st.selectbox(
            "🌍 Paese",
            paesi,
            index=paesi.index("IT") if "IT" in paesi else 0,
            key="val_paese",
        )
    with fc2:
        val_settore = st.selectbox(
            "🏭 Settore NACE",
            settori,
            format_func=nace_label,
            key="val_settore",
        )
        if val_settore in NACE_LABELS:
            st.caption(f"ℹ️ {NACE_LABELS[val_settore]}")
    with fc3:
        val_anno = st.number_input(
            "📅 Anno di riferimento",
            min_value=2000, max_value=2035,
            value=int(max(anni)), key="val_anno",
        )

    fv1, fv2 = st.columns(2)
    with fv1:
        val_indicatore = st.selectbox(
            "📌 Indicatore da analizzare",
            INDICATORS,
            format_func=lambda x: f"{x} – {IND_LABELS[x]}",
            key="val_ind",
        )
        st.caption(f"ℹ️ {IND_DESCRIPTIONS[val_indicatore]}")
    with fv2:
        val_reale = st.number_input(
            f"💰 Valore reale azienda  ({IND_LABELS.get(val_indicatore, '')})",
            min_value=0.0, value=1000.0, step=10.0,
            format="%.2f", key="val_reale",
        )

    st.markdown("")
    btn_valuta = st.button("▶ Valuta l'azienda", type="primary",
                           use_container_width=False)

    if btn_valuta:
        # ── Carica modelli ─────────────────────────────────────────────────────
        pay_v = carica_modello(val_indicatore)
        kd_v  = carica_kmeans(val_indicatore)

        if pay_v is None:
            st.error("Modello ML non trovato. Avvia prima il training dalla sidebar.")
        else:
            # ── Previsione benchmark ML ────────────────────────────────────────
            Xnew = pd.DataFrame({
                "nace_r2":     [val_settore],
                "geo":         [val_paese],
                "TIME_PERIOD": [val_anno],
            })
            try:
                benchmark = float(pay_v["model"].predict(Xnew)[0])
            except Exception as ex:
                st.error(f"Errore nella previsione: {ex}")
                st.stop()

            gap_abs = val_reale - benchmark
            gap_pct = (gap_abs / benchmark * 100) if benchmark != 0 else 0.0

            # ── Z-score e profilo clustering ───────────────────────────────────
            zscore_val   = None
            profilo_val  = "N/D"
            mu_val       = None
            sigma_val    = None

            if kd_v is not None:
                try:
                    if "sector_stats" in kd_v and val_settore in kd_v["sector_stats"].index:
                        mu_val    = float(kd_v["sector_stats"].loc[val_settore, "mu"])
                        sigma_val = float(kd_v["sector_stats"].loc[val_settore, "sigma"])
                    else:
                        mu_val    = float(df_full[
                            (df_full["indic_sb"] == val_indicatore)
                        ]["OBS_VALUE"].median())
                        sigma_val = 1.0
                    sigma_val = max(sigma_val, 1e-6)
                    zscore_val  = float(np.clip(
                        (val_reale - mu_val) / sigma_val, -3, 3))
                    Xcl_v = kd_v["scaler"].transform([[zscore_val, val_anno]])
                    lbl_v = int(kd_v["km"].predict(Xcl_v)[0])
                    profilo_val = kd_v["label_map"].get(lbl_v, "N/D")
                except Exception:
                    pass

            # ── Output sezione KPI ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Risultati della valutazione")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Benchmark ML (previsto)",
                f"{benchmark:,.2f}",
                help=f"Valore atteso dal modello {pay_v['name']}",
            )
            k2.metric(
                "Valore reale inserito",
                f"{val_reale:,.2f}",
            )
            k3.metric(
                "Gap assoluto",
                f"{gap_abs:+,.2f}",
                delta=f"{gap_pct:+.1f}%",
                delta_color="normal",
            )
            if zscore_val is not None:
                k4.metric(
                    "Z-score settore",
                    f"{zscore_val:+.2f} σ",
                    help="Scostamento dalla media storica del settore in deviazioni standard",
                )

            # ── Badge profilo ──────────────────────────────────────────────────
            css_map = {
                "Alta Performance":  "risk-alta",
                "Media Performance": "risk-media",
                "Bassa Performance": "risk-bassa",
            }
            css_badge = css_map.get(profilo_val, "risk-media")
            st.markdown(
                f"**Profilo di performance:**&nbsp;&nbsp;"
                f'<span class="{css_badge}">{profilo_val}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            # ── Grafici: gauge + posizionamento ───────────────────────────────
            col_g, col_p = st.columns([1.1, 1.9])

            # Gauge chart – z-score
            with col_g:
                st.markdown("**Posizionamento z-score (±3σ)**")
                zv = zscore_val if zscore_val is not None else 0.0
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=zv,
                    delta={"reference": 0, "suffix": " σ",
                           "valueformat": "+.2f"},
                    number={"suffix": " σ", "valueformat": "+.2f"},
                    gauge={
                        "axis": {"range": [-3, 3], "tickwidth": 1,
                                 "tickcolor": "#555",
                                 "tickvals": [-3, -2, -1, 0, 1, 2, 3]},
                        "bar":  {"color": (
                            "#27ae60" if zv >= 0.5
                            else "#e74c3c" if zv <= -0.5
                            else "#f39c12")},
                        "bgcolor": "white",
                        "steps": [
                            {"range": [-3, -0.5], "color": "#fde8e8"},
                            {"range": [-0.5, 0.5], "color": "#fef9e7"},
                            {"range": [0.5,  3],   "color": "#eafaf1"},
                        ],
                        "threshold": {
                            "line": {"color": "#2c3e50", "width": 3},
                            "thickness": 0.75,
                            "value": zv,
                        },
                    },
                    title={"text": "Z-score vs media settore"},
                ))
                fig_gauge.update_layout(
                    height=280, margin=dict(l=20, r=20, t=50, b=10),
                    paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Grafico posizionamento rispetto alla distribuzione del settore
            with col_p:
                st.markdown("**Posizionamento vs distribuzione settore**")
                df_settore = df_full[
                    (df_full["indic_sb"] == val_indicatore) &
                    (df_full["nace_r2"] == val_settore)
                ]["OBS_VALUE"].dropna()

                if len(df_settore) >= 5:
                    fig_pos = go.Figure()
                    fig_pos.add_trace(go.Violin(
                        y=df_settore,
                        name="Distribuzione settore",
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor="#aed6f1",
                        opacity=0.6,
                        line_color="#1a5276",
                        points="outliers",
                    ))
                    # Linea valore reale
                    fig_pos.add_hline(
                        y=val_reale, line_dash="dash", line_color="#e74c3c",
                        line_width=2,
                        annotation_text=f"La tua azienda: {val_reale:,.0f}",
                        annotation_position="top right",
                        annotation_font_color="#e74c3c",
                    )
                    # Linea benchmark
                    fig_pos.add_hline(
                        y=benchmark, line_dash="dot", line_color="#27ae60",
                        line_width=2,
                        annotation_text=f"Benchmark ML: {benchmark:,.0f}",
                        annotation_position="bottom right",
                        annotation_font_color="#27ae60",
                    )
                    fig_pos.update_layout(
                        height=280,
                        margin=dict(l=0, r=0, t=20, b=0),
                        yaxis_title=IND_LABELS[val_indicatore],
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
                else:
                    st.info("Dati insufficienti per la distribuzione del settore.")

            # ── Tabella riepilogativa ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Riepilogo analitico")

            righe_riepilogo = {
                "Paese":                     val_paese,
                "Settore NACE":              f"{val_settore} – {NACE_LABELS.get(val_settore, '')}",
                "Anno":                      str(val_anno),
                "Indicatore":                f"{val_indicatore} – {IND_LABELS[val_indicatore]}",
                "Valore reale (inserito)":   f"{val_reale:,.2f}",
                "Benchmark ML":              f"{benchmark:,.2f}",
                "Modello utilizzato":        pay_v["name"],
                "Gap assoluto":              f"{gap_abs:+,.2f}",
                "Gap percentuale":           f"{gap_pct:+.1f}%",
                "Z-score settore":           (f"{zscore_val:+.2f} σ"
                                              if zscore_val is not None else "N/D"),
                "Media storica settore":     (f"{mu_val:,.2f}"
                                              if mu_val is not None else "N/D"),
                "Dev. std settore (σ)":      (f"{sigma_val:,.2f}"
                                              if sigma_val is not None else "N/D"),
                "Profilo di performance":    profilo_val,
            }
            df_riepilogo = pd.DataFrame(
                list(righe_riepilogo.items()),
                columns=["Parametro", "Valore"],
            )
            st.dataframe(df_riepilogo, use_container_width=True, hide_index=True)

            # ── Suggerimento manageriale ───────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💡 Interpretazione manageriale")

            if profilo_val == "Alta Performance":
                st.success(
                    f"L'azienda supera il benchmark di settore del **{gap_pct:+.1f}%** "
                    f"e si colloca in una zona di **alta performance** "
                    f"({zscore_val:+.2f}σ sopra la media). "
                    "Questo risultato suggerisce un vantaggio competitivo strutturale "
                    "o un periodo di crescita sostenuta."
                )
            elif profilo_val == "Media Performance":
                st.info(
                    f"L'azienda è allineata con la media del settore "
                    f"(gap: **{gap_pct:+.1f}%**, z-score: **{zscore_val:+.2f}σ**). "
                    "Le performance sono in linea con il benchmark, con margini "
                    "di miglioramento identificabili attraverso le leve operative."
                )
            else:
                st.warning(
                    f"L'azienda si trova **al di sotto del benchmark** "
                    f"di **{abs(gap_pct):.1f}%** (z-score: **{zscore_val:+.2f}σ**). "
                    "Questo segnale di early warning suggerisce di analizzare "
                    "i driver di costo o le cause di sottoperformance rispetto "
                    f"alla media del settore **{val_settore}**."
                )

            st.caption(
                f"Analisi effettuata con modello {pay_v['name']} | "
                f"Anno cutoff training: {pay_v.get('anno_cutoff', 'N/D')} | "
                "Fonte dati: Eurostat SBS (PMI)"
            )
