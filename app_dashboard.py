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

# Indicatori di performance usati in tutti i tab (modelli ML, clustering, confronto)
INDICATORS = ["V12110", "V12120", "V12130", "V12150"]
# V11110 (Numero di imprese) è un dato strutturale, non di performance:
# non viene modellato né mostrato nei tab ML — ha la sua sezione in Tab 4.
IND_V11110 = "V11110"
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
# ORIENTAMENTO DEGLI INDICATORI (per Tab 5 – Valutazione Aziendale)
# "positive" = più alto è meglio (ricavi, valore aggiunto, produzione)
# "cost"     = più alto = costi maggiori → interpretazione INVERTITA
# "count"    = conteggio settoriale, non metrica di performance individuale
# ─────────────────────────────────────────────────────────────────────────────
INDICATOR_ORIENTATION = {
    "V11110": "count",
    "V12110": "positive",
    "V12120": "positive",
    "V12130": "positive",
    "V12150": "cost",
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
    # ── Aggregati Eurostat SBS presenti nel dataset (non codici NACE standard) ──
    "B-N_S95_X_K":   "Economia d'impresa (industria, costruzioni e servizi, escluse attività finanziarie)",
    "KIA":            "Attività ad alta intensità di conoscenza",
    "KIABI_X_K_R90":  "Servizi knowledge-intensive alle imprese (escluse finanza e attività ricreative)",
}

# ── Nomi completi dei paesi (codici Eurostat → italiano) ──────────────────────
GEO_LABELS: dict[str, str] = {
    "AL":       "Albania",
    "AT":       "Austria",
    "BA":       "Bosnia ed Erzegovina",
    "BE":       "Belgio",
    "BG":       "Bulgaria",
    "CH":       "Svizzera",
    "CY":       "Cipro",
    "CZ":       "Repubblica Ceca",
    "DE":       "Germania",
    "DK":       "Danimarca",
    "EE":       "Estonia",
    "EL":       "Grecia",
    "ES":       "Spagna",
    "EU27_2020":"UE-27 (aggregato 2020)",
    "EU28":     "UE-28 (aggregato pre-Brexit)",
    "FI":       "Finlandia",
    "FR":       "Francia",
    "HR":       "Croazia",
    "HU":       "Ungheria",
    "IE":       "Irlanda",
    "IS":       "Islanda",
    "IT":       "Italia",
    "LT":       "Lituania",
    "LU":       "Lussemburgo",
    "LV":       "Lettonia",
    "MK":       "Macedonia del Nord",
    "MT":       "Malta",
    "NL":       "Paesi Bassi",
    "NO":       "Norvegia",
    "PL":       "Polonia",
    "PT":       "Portogallo",
    "RO":       "Romania",
    "RS":       "Serbia",
    "SE":       "Svezia",
    "SI":       "Slovenia",
    "SK":       "Slovacchia",
    "TR":       "Turchia",
    "UK":       "Regno Unito",
}

def geo_label(code: str) -> str:
    """Restituisce 'CODICE – Nome paese' oppure solo 'CODICE' se non trovato."""
    nome = GEO_LABELS.get(code, "")
    return f"{code} – {nome}" if nome else code

def nace_label(code: str) -> str:
    """Restituisce 'Descrizione (CODICE)' oppure solo 'CODICE' se descrizione non trovata."""
    desc = NACE_LABELS.get(code, "")
    return f"{desc} ({code})" if desc else code
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

@st.cache_data
def stats_settore_indicatore(indicatore: str, settore: str):
    """
    Statistiche di distribuzione storica per (indicatore, settore NACE).

    Restituisce due livelli di statistiche:
    1. Aggregato di settore (valori Eurostat SBS come presenti nel dataset)
    2. Per-impresa (aggregato diviso per V11110 – numero di imprese)
       disponibile per tutti gli indicatori monetari (non per V11110 stesso).

    Usato nel form Tab 5 per mostrare range realistici a livello di singola PMI.
    """
    df = carica_dati()
    df_ind = df[(df["indic_sb"] == indicatore) & (df["nace_r2"] == settore)].copy()
    if len(df_ind) < 3:
        return None

    vals = df_ind["OBS_VALUE"].dropna()
    res = dict(
        p5      = float(vals.quantile(0.05)),
        p25     = float(vals.quantile(0.25)),
        mediana = float(vals.median()),
        media   = float(vals.mean()),
        p75     = float(vals.quantile(0.75)),
        p95     = float(vals.quantile(0.95)),
        std     = float(vals.std()),
        n       = int(len(vals)),
    )

    # ── Stima per-impresa (solo per indicatori monetari) ──────────────────────
    if indicatore != "V11110":
        df_n = df[(df["indic_sb"] == "V11110") & (df["nace_r2"] == settore)][
            ["geo", "TIME_PERIOD", "OBS_VALUE"]
        ].rename(columns={"OBS_VALUE": "n_ent"})

        if len(df_n) > 0:
            df_m = df_ind.merge(df_n, on=["geo", "TIME_PERIOD"], how="left")
            df_m["n_ent"] = pd.to_numeric(df_m["n_ent"], errors="coerce").replace(0, np.nan)
            df_m["val_pe"] = df_m["OBS_VALUE"] / df_m["n_ent"]
            pe = df_m["val_pe"].dropna()
            if len(pe) >= 3:
                res.update(dict(
                    pe_p5      = float(pe.quantile(0.05)),
                    pe_p25     = float(pe.quantile(0.25)),
                    pe_mediana = float(pe.median()),
                    pe_p75     = float(pe.quantile(0.75)),
                    pe_p95     = float(pe.quantile(0.95)),
                    pe_std     = float(pe.std()),
                    pe_n       = int(len(pe)),
                ))
    return res


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
        format_func=geo_label,
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
    # ── KPI rapido: ultimo valore disponibile per paese+indicatore ───────────
    _df_kpi = df_full[(df_full["indic_sb"] == ind_sel) & (df_full["geo"] == paese)]
    if len(_df_kpi):
        _kpi_anno = int(_df_kpi["TIME_PERIOD"].max())
        _kpi_vals = _df_kpi[_df_kpi["TIME_PERIOD"] == _kpi_anno]["OBS_VALUE"].dropna()
        if len(_kpi_vals):
            _kpi_val = float(_kpi_vals.mean())
            # Calcola delta rispetto all'anno precedente
            _df_kpi_prev = _df_kpi[_df_kpi["TIME_PERIOD"] == _kpi_anno - 1]["OBS_VALUE"].dropna()
            _kpi_delta = None
            if len(_df_kpi_prev):
                _kpi_delta = f"{((_kpi_val - float(_df_kpi_prev.mean())) / abs(float(_df_kpi_prev.mean())) * 100):+.1f}% vs {_kpi_anno-1}"
            st.metric(
                f"📊 {IND_LABELS[ind_sel].replace(' (M€)','')} – {_kpi_anno}",
                f"{_kpi_val:,.0f} M€",
                delta=_kpi_delta,
                help=f"Valore più recente di {IND_LABELS[ind_sel]} per {GEO_LABELS.get(paese, paese)} (fonte Eurostat SBS)."
            )
    st.markdown("---")
    pay0 = carica_modello(INDICATORS[0])
    if pay0 and "anno_cutoff" in pay0:
        ac = pay0["anno_cutoff"]
        st.caption(f"Train ≤{ac} | Test >{ac} | CV {N_CV_SPLITS} fold")
    st.caption("Tesi LM31 · Università Mercatorum")

# ── HEADER + KPI ──────────────────────────────────────────────────────────────
st.markdown("# Dashboard Interattiva – PMI Europee (Eurostat SBS)")
st.markdown(
    f"**Indicatore:** `{ind_sel}` – *{IND_LABELS[ind_sel]}*  |  "
    f"**Paese:** {geo_label(paese)}  |  **Settore:** {nace_label(settore)}"
)
st.caption(f"📊 {IND_DESCRIPTIONS[ind_sel]}")
st.markdown("---")

df_ind = df_full[df_full["indic_sb"] == ind_sel]
df_fil = df_ind[(df_ind["geo"] == paese) & (df_ind["nace_r2"] == settore)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Osservazioni filtrate", len(df_fil),
          help="Righe nel dataset Eurostat SBS per la combinazione Paese + Settore NACE selezionata nella sidebar. "
               "Se il valore è 0 significa che Eurostat non ha raccolto dati per quella combinazione.")
c2.metric("Media storica",
          f"{df_fil['OBS_VALUE'].mean():,.0f}" if len(df_fil) else "N/D",
          help="Media aritmetica dei valori storici dell'indicatore selezionato, per il paese e settore correnti. "
               "I valori sono espressi in milioni di euro (M€).")
c3.metric("Paesi nel dataset", df_ind["geo"].nunique(),
          help="Numero di paesi europei (EU + candidati) per cui Eurostat ha fornito dati "
               "per l'indicatore selezionato. Include sia singoli stati che aggregati EU27/EU28.")
c4.metric("Settori nel dataset", df_ind["nace_r2"].nunique(),
          help="Numero di settori NACE Rev.2 per cui l'indicatore è disponibile nel dataset. "
               "Il dataset SBS copre 4 macrosettori principali: B, B-N_S95_X_K, KIA, KIABI_X_K_R90.")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Previsione ML", "Early Warning", "Confronto Modelli",
    "Esplorazione Dataset", "🏢 Valutazione Aziendale"])


# ══ TAB 1 – PREVISIONE ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("### Previsione con il modello ML")
    with st.expander("📖 Come funziona la previsione ML", expanded=False):
        st.markdown("""
**Cosa fa questo tab?**
Il modello di Machine Learning è stato addestrato sui dati storici Eurostat SBS (2011–2020) e
impara a stimare il valore atteso di un indicatore economico in funzione di tre variabili:
**paese**, **settore NACE** e **anno**. Inserendo questi parametri nel form,
il modello restituisce la previsione per una combinazione mai vista.

**I modelli disponibili:**
- **Ridge Regression** – modello lineare regolarizzato, robusto alle correlazioni tra variabili
- **Random Forest** – ensemble di alberi decisionali, cattura relazioni non lineari
- **Gradient Boosting** – apprendimento sequenziale degli errori, spesso il più accurato
- **MLP (Rete Neurale)** – percettrone multi-strato, adatto a pattern complessi

Il sistema seleziona automaticamente il modello con il miglior **R² sul test set** per ciascun indicatore.

**Come leggere le metriche di accuratezza:**
- **MAE** *(Mean Absolute Error)*: errore medio in valore assoluto. Più basso = più preciso. Espresso nella stessa unità dell'indicatore (M€).
- **RMSE** *(Root Mean Squared Error)*: simile al MAE ma penalizza gli errori grandi. Se RMSE >> MAE, il modello fa errori gravi su alcuni casi.
- **R²** *(Coefficiente di determinazione)*: percentuale di variabilità dei dati spiegata dal modello. R²=1 = perfetto; R²=0 = equivalente a prevedere sempre la media.

**Profilo di performance:** dopo la previsione, il valore viene classificato con K-Means (k=3) rispetto
alla media storica del settore, restituendo: 🟢 Alta / 🟡 Media / 🔴 Bassa Performance.
""")

    pay = carica_modello(ind_sel)
    if pay is None:
        st.warning("Modello non trovato. Riavvia la dashboard.")
    else:
        col1, col2 = st.columns([1, 1.8])
        with col1:
            st.markdown(f"**Modello attivo:** `{pay['name']}`")
            m = pay["metriche"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE",     f"{m['MAE']:,.2f}",
                       help="Mean Absolute Error: errore medio assoluto in M€. "
                            "Indica di quanto si discosta mediamente la previsione dal valore reale. Più basso = più preciso.")
            mc2.metric("RMSE",    f"{m['RMSE']:,.2f}",
                       help="Root Mean Squared Error: come il MAE ma penalizza gli errori grandi. "
                            "Se RMSE è molto maggiore del MAE il modello fa errori gravi su alcuni casi specifici.")
            mc3.metric("R² test", f"{m['R2']:.4f}",
                       help="Coefficiente di determinazione sul test set (anni più recenti). "
                            "Valori vicini a 1.0 indicano un ottimo fit; valori negativi indicano che il modello "
                            "performa peggio di una semplice media storica.")

            st.markdown("---")
            st.markdown("#### Inserisci i parametri")
            p_in = st.selectbox("Paese", paesi,
                                index=paesi.index(paese),
                                format_func=geo_label, key="p1")
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
            # ── Dati filtrati sui valori del FORM (p_in / s_in / a_in) ──────────
            df_fil_t1 = df_full[
                (df_full["indic_sb"] == ind_sel) &
                (df_full["geo"]      == p_in) &
                (df_full["nace_r2"]  == s_in)
            ]

            st.markdown(
                f"#### Serie storica – {GEO_LABELS.get(p_in, p_in)} · "
                f"{NACE_LABELS.get(s_in, s_in)}"
            )
            if len(df_fil_t1):
                _df_t1_sorted = df_fil_t1.sort_values("TIME_PERIOD")
                fig = px.line(_df_t1_sorted,
                              x="TIME_PERIOD", y="OBS_VALUE", markers=True,
                              color_discrete_sequence=["#2980b9"],
                              labels={"TIME_PERIOD": "Anno",
                                      "OBS_VALUE": IND_LABELS[ind_sel]})
                # Linea verticale tratteggiata sull'anno selezionato (se nel range)
                _anni_t1 = sorted(df_fil_t1["TIME_PERIOD"].unique())
                if int(a_in) in _anni_t1:
                    fig.add_vline(
                        x=int(a_in), line_dash="dash", line_color="#e74c3c",
                        annotation_text=f"Anno selezionato: {int(a_in)}",
                        annotation_position="top left",
                        annotation_font_color="#e74c3c",
                    )
                fig.update_layout(height=310, margin=dict(l=0, r=0, t=30, b=0),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True, key="pc_t1_serie")
                # Insight automatico: variazione percentuale nel periodo
                _df_s = _df_t1_sorted.dropna(subset=["OBS_VALUE"])
                if len(_df_s) >= 2:
                    _v_ini = float(_df_s["OBS_VALUE"].iloc[0])
                    _v_fin = float(_df_s["OBS_VALUE"].iloc[-1])
                    _a_ini = int(_df_s["TIME_PERIOD"].iloc[0])
                    _a_fin = int(_df_s["TIME_PERIOD"].iloc[-1])
                    if _v_ini != 0:
                        _var_pct = (_v_fin - _v_ini) / abs(_v_ini) * 100
                        _icon    = "📈" if _var_pct > 5 else ("📉" if _var_pct < -5 else "➡️")
                        _trend_t = "in crescita" if _var_pct > 5 else ("in calo" if _var_pct < -5 else "stabile")
                        st.info(
                            f"{_icon} **Trend {_a_ini}→{_a_fin}**: **{_trend_t}** "
                            f"(**{_var_pct:+.1f}%**, da {_v_ini:,.0f} a {_v_fin:,.0f} M€)"
                        )
                st.caption(
                    f"Andamento storico di **{IND_LABELS[ind_sel]}** "
                    f"per {geo_label(p_in)} – {nace_label(s_in)} (fonte: Eurostat SBS). "
                    "La linea rossa indica l'anno selezionato nel form."
                )
            else:
                st.info(
                    f"Nessun dato storico per {geo_label(p_in)} – {nace_label(s_in)}. "
                    "Prova a cambiare paese o settore nel form a sinistra."
                )

            # ── Classifica paesi per l'anno e l'indicatore del form ─────────────
            # Filtra per indicatore del form; usa l'anno del form con fallback al più vicino
            _df_cls   = df_full[(df_full["indic_sb"] == ind_sel)]
            _anni_cls = sorted(_df_cls["TIME_PERIOD"].unique())
            _anno_cls = int(a_in)
            if _anno_cls > max(_anni_cls):
                _anno_cls = int(max(_anni_cls))
            elif _anno_cls < min(_anni_cls):
                _anno_cls = int(min(_anni_cls))

            st.markdown(f"#### Classifica paesi – {_anno_cls}")
            if _anno_cls != int(a_in):
                st.caption(
                    f"ℹ️ Anno {int(a_in)} fuori dal dataset – "
                    f"visualizzato l'anno più vicino: **{_anno_cls}**."
                )
            dm = (_df_cls[_df_cls["TIME_PERIOD"] == _anno_cls]
                  .groupby("geo")["OBS_VALUE"].mean().reset_index())
            dm["NomePaese"] = dm["geo"].map(lambda c: GEO_LABELS.get(c, c))
            _dm_top  = dm.sort_values("OBS_VALUE", ascending=False).head(15).copy()
            _dm_plot = _dm_top.sort_values("OBS_VALUE", ascending=True)
            fig2 = px.bar(_dm_plot, y="NomePaese", x="OBS_VALUE", orientation="h",
                          color="OBS_VALUE", color_continuous_scale="Blues",
                          labels={"NomePaese": "", "OBS_VALUE": IND_LABELS[ind_sel]})
            # Evidenzia il paese del form con linea rossa tratteggiata
            _paese_lbl = GEO_LABELS.get(p_in, p_in)
            if _paese_lbl in _dm_plot["NomePaese"].values:
                fig2.add_shape(
                    type="line", xref="paper", x0=0, x1=1,
                    yref="y", y0=_paese_lbl, y1=_paese_lbl,
                    line=dict(color="#e74c3c", width=2, dash="dot"),
                )
                fig2.add_annotation(
                    xref="paper", x=1.01, yref="y", y=_paese_lbl,
                    text=_paese_lbl, showarrow=False,
                    font=dict(color="#e74c3c", size=11), xanchor="left",
                )
            fig2.update_layout(height=400, margin=dict(l=0, r=80, t=10, b=0),
                               coloraxis_showscale=False,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True, key="pc_t1_classifica")
            # Insight: posizione del paese selezionato
            _rank_df    = _dm_top.sort_values("OBS_VALUE", ascending=False).reset_index(drop=True)
            _rank_paese = _rank_df[_rank_df["geo"] == p_in]
            if len(_rank_paese):
                _pos = int(_rank_paese.index[0]) + 1
                _tot = len(_rank_df)
                st.info(
                    f"💡 **{GEO_LABELS.get(p_in, p_in)}** è al **{_pos}° posto** "
                    f"su {_tot} paesi per **{IND_LABELS[ind_sel]}** nell'anno {_anno_cls}."
                )
            st.caption(
                f"🌍 Classifica dei primi 15 paesi per **{IND_LABELS[ind_sel]}** nell'anno **{_anno_cls}**. "
                "Si aggiorna automaticamente al cambio di anno, paese o indicatore."
            )


# ══ TAB 2 – EARLY WARNING ════════════════════════════════════════════════════
with tab2:
    st.markdown("### Early Warning – Profili di Performance")
    with st.expander("📖 Come funziona l'Early Warning", expanded=False):
        st.markdown("""
**Cosa fa questo tab?**
Il sistema di Early Warning individua automaticamente i **profili di rischio** delle PMI europee
tramite un algoritmo di clustering non supervisionato (**K-Means, k=3**).
L'obiettivo è classificare ogni osservazione in uno dei tre profili:
- 🟢 **Alta Performance** – valori significativamente sopra la media del settore
- 🟡 **Media Performance** – valori allineati alla media del settore
- 🔴 **Bassa Performance** – valori significativamente sotto la media del settore

**Come funziona il K-Means?**
L'algoritmo raggruppa le osservazioni in 3 cluster minimizzando la distanza interna a ciascun gruppo.
Ogni PMI viene assegnata al cluster il cui centroide (valore medio) è più vicino al suo valore normalizzato.

**Normalizzazione Z-score per settore:**
Prima del clustering ogni valore viene trasformato in z-score: *z = (valore − media_settore) / std_settore*.
Questo permette di confrontare PMI dello stesso settore su scala relativa, eliminando le differenze
di grandezza tra settori diversi (es. manifatturiero vs. estrattivo).

**Silhouette Score:**
Misura quanto ogni punto è simile al proprio cluster rispetto agli altri.
- Valori vicini a **+1** → cluster ben separati e coesi
- Valori vicini a **0** → cluster sovrapposti
- Valori negativi → possibili assegnazioni errate

**Importante:** i parametri del clustering (centroidi, scaler) sono calcolati **solo sul training set**,
garantendo che il test set non influenzi l'addestramento (*no data leakage*).
""")

    kd = carica_kmeans(ind_sel)
    if kd is None:
        st.warning("Modello K-Means non trovato. Riavvia la dashboard.")
    else:
        sil = kd.get("silhouette", None)
        ki1, ki2, ki3 = st.columns(3)
        ki1.metric("Algoritmo", "K-Means  (k=3)",
                   help="K-Means con k=3 cluster: Alta / Media / Bassa Performance. "
                        "Addestrato solo sul training set per evitare data leakage.")
        ki2.metric("Silhouette Score",
                   f"{sil:.3f}" if sil is not None else "N/D",
                   help="Misura la qualità della separazione tra cluster (range −1 → +1). "
                        "Sopra 0.4 = buona separazione; sotto 0.2 = cluster sovrapposti.")
        ki3.metric("Normalizzazione", "Z-score per settore",
                   help="Ogni valore viene standardizzato rispetto alla media e deviazione standard "
                        "del proprio settore NACE, per rendere comparabili settori di dimensioni diverse.")

        if sil is not None:
            q = ("Buona separazione" if sil > 0.4
                 else "Separazione accettabile" if sil > 0.2
                 else "Cluster sovrapposti")
            st.caption(f"Qualità clustering: {q} (silhouette = {sil:.3f})")

        st.info(
            "Il profilo di performance è calcolato tramite z-score rispetto "
            "alla media storica del settore (nace_r2). Una PMI manifatturiera "
            "viene confrontata solo con altre PMI del manifatturiero, non con settori diversi.")

        # Aggregati EU da escludere dalla visualizzazione:
        # EU27_2020 e EU28 sono somme di tutti i paesi → valori 10-20x superiori
        # ai singoli stati → distorcono la scala e non rappresentano singole PMI.
        # Vengono tenuti nel dataset per i calcoli ML ma esclusi dalla visualizzazione.
        _EU_EXCL = {"EU27_2020", "EU28"}

        # Clustering con z-score (su tutti i dati, aggregati EU inclusi)
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

        # Vista per visualizzazione: solo singoli paesi (no aggregati EU)
        df_cl_viz = df_cl[~df_cl["geo"].isin(_EU_EXCL)].copy()

        st.caption(
            "ℹ️ Gli aggregati **EU27** e **EU28** sono esclusi dai grafici: "
            "rappresentano la somma di tutti i paesi europei e non singole PMI, "
            "renderebbero la scala del grafico illeggibile."
        )

        c1, c2 = st.columns([1, 1.6])
        with c1:
            # Distribuzione profili calcolata sui soli paesi singoli
            dist = df_cl_viz["profilo"].value_counts().reset_index()
            dist.columns = ["Profilo", "N"]
            dist["Percentuale"] = (dist["N"] / len(df_cl_viz) * 100).round(1)
            fig_p = px.pie(dist, names="Profilo", values="N",
                           color="Profilo",
                           color_discrete_map=RISK_COLORS, hole=0.4)
            fig_p.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_p, use_container_width=True, key="pc_t2_pie")
            st.caption(
                "🍩 Distribuzione percentuale dei 3 profili K-Means "
                "(singoli paesi europei, aggregati EU esclusi)."
            )
            st.dataframe(dist.style.format({"Percentuale": "{:.1f}%"}),
                         use_container_width=True, hide_index=True)
            # Insight automatico sul clustering
            _ap = dist[dist["Profilo"] == "Alta Performance"]["Percentuale"].values
            _bp = dist[dist["Profilo"] == "Bassa Performance"]["Percentuale"].values
            if len(_ap) and len(_bp):
                _ap_v, _bp_v = float(_ap[0]), float(_bp[0])
                _dominant = ("Alta" if _ap_v > _bp_v else "Bassa")
                _icon_ew = "🟢" if _dominant == "Alta" else "🔴"
                st.info(
                    f"{_icon_ew} **Insight Early Warning** – {IND_LABELS[ind_sel]}: "
                    f"il **{_ap_v:.0f}%** dei paesi è in Alta Performance, "
                    f"il **{_bp_v:.0f}%** è in Bassa Performance. "
                    + ("Il settore mostra segnali prevalentemente positivi."
                       if _dominant == "Alta"
                       else "Il settore mostra segnali di difficoltà diffusa.")
                )
        with c2:
            _df_sc = df_cl_viz.sort_values("profilo").copy()
            _df_sc["Paese"]   = _df_sc["geo"].map(geo_label)
            _df_sc["Settore"] = _df_sc["nace_r2"].map(nace_label)
            fig_sc = px.scatter(
                _df_sc,
                x="TIME_PERIOD", y="OBS_VALUE", color="profilo",
                color_discrete_map=RISK_COLORS,
                hover_data=["Paese", "Settore"], opacity=0.65,
                labels={"TIME_PERIOD": "Anno",
                        "OBS_VALUE":   IND_LABELS[ind_sel],
                        "profilo":     "Profilo"})
            fig_sc.update_traces(marker_size=7)
            fig_sc.update_layout(height=370, margin=dict(l=0, r=0, t=20, b=0),
                                 legend=dict(orientation="h",
                                             yanchor="bottom", y=1.02),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sc, use_container_width=True, key="pc_t2_scatter")
            st.caption(
                "📊 Ogni punto = un paese in un anno. "
                "Colore = profilo K-Means. "
                "Passa il mouse per vedere paese e settore. "
                "Cluster ben separati confermano che il modello distingue correttamente le performance."
            )

        st.markdown("---")
        prof_s = st.radio("Filtra per profilo:",
                          ["Bassa Performance", "Media Performance",
                           "Alta Performance"],
                          horizontal=True)
        # Grafico a barre: usa df_cl_viz (no EU aggregati) con nomi paese leggibili
        dr = (df_cl_viz[df_cl_viz["profilo"] == prof_s]
              .groupby("geo")["OBS_VALUE"]
              .agg(["mean", "count"]).reset_index()
              .rename(columns={"geo": "Codice", "mean": "Media", "count": "N obs"}))
        dr["NomePaese"] = dr["Codice"].map(lambda c: GEO_LABELS.get(c, c))
        dr["Media"] = dr["Media"].round(2)
        dr = dr.sort_values("Media", ascending=(prof_s == "Bassa Performance"))
        if len(dr):
            fig_r = px.bar(
                dr.head(20).sort_values("Media", ascending=True),
                y="NomePaese", x="Media",
                orientation="h",
                color_discrete_sequence=[RISK_COLORS[prof_s]],
                labels={"NomePaese": "", "Media": IND_LABELS[ind_sel]})
            fig_r.update_layout(height=350, margin=dict(l=0, r=20, t=10, b=0),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_r, use_container_width=True, key="pc_t2_bar")
            st.caption(
                f"🌍 Paesi in **{prof_s}** per {IND_LABELS[ind_sel]}, ordinati per valore medio. "
                "Cambia il profilo sopra per confrontare i paesi nelle tre fasce."
            )


# ══ TAB 3 – CONFRONTO MODELLI ════════════════════════════════════════════════
with tab3:
    st.markdown("### Confronto performance modelli ML")
    with st.expander("📖 Come leggere il confronto tra modelli", expanded=False):
        st.markdown("""
**Cosa fa questo tab?**
Confronta le prestazioni di 4 algoritmi di Machine Learning + 1 modello Baseline sullo stesso dataset,
per ciascuno dei 5 indicatori Eurostat SBS. L'obiettivo è identificare quale modello generalizza meglio
su dati *mai visti* durante l'addestramento.

**Metodologia di valutazione:**
- **Test set temporale**: il dataset è diviso cronologicamente. Gli anni più recenti (~20%) servono
  solo per la valutazione finale e non sono mai usati per l'addestramento. Questo simula l'uso reale
  del modello su dati futuri.
- **5-fold TimeSeriesSplit**: durante l'addestramento si usa la cross-validation temporale per stimare
  la stabilità del modello su finestre di anni diverse.

**Il modello Baseline:**
È un punto di riferimento semplice: prevede la mediana storica per ogni combinazione paese-settore.
Un modello ML è utile *solo se supera* il Baseline, altrimenti non aggiunge valore rispetto a una
semplice media storica.

**Metriche in tabella:**
| Metrica | Significato | Meglio se |
|---------|-------------|-----------|
| **MAE** | Errore medio assoluto in M€ | Più basso |
| **RMSE** | Come MAE ma penalizza errori grandi | Più basso |
| **R²** | % variabilità spiegata (0-1) | Più vicino a 1 |
| **R²_CV_media** | R² medio nelle 5 fold temporali | Più vicino a 1 |

**Feature Importance:**
Indica quali variabili il modello considera più rilevanti per fare previsioni.
Le 3 feature usate sono: **Settore NACE** (codificato), **Paese** (codificato), **Anno**.
""")

    df_conf = carica_confronto()
    all_res = carica_risultati()
    fi_data = carica_fi()

    if df_conf is None or all_res is None:
        st.warning("File risultati non trovati. Riavvia la dashboard.")
    else:
        st.info(
            "Metriche calcolate su **test set temporale** (anni più recenti) "
            "e tramite **5-fold TimeSeriesSplit** sul training set. "
            "Il **Baseline** è la mediana storica per paese-settore: "
            "un modello ML è utile solo se lo supera.")

        df_ci = df_conf[df_conf["Indicatore"] == ind_sel].copy()

        # ── Riquadro: criteri di scelta del modello ────────────────────────────
        with st.expander("📐 Criteri tecnici di selezione del modello", expanded=False):
            st.markdown("""
### Come viene scelto il modello migliore?

Il sistema seleziona automaticamente il modello con il **miglior R² sul test set** per ciascun indicatore.
L'R² è il criterio principale perché misura la **capacità di generalizzazione** del modello su dati mai visti.

---

### Perché l'R² sul test set, e non il MAE?

| Criterio | Cosa misura | Limite |
|----------|-------------|--------|
| **MAE** | Errore medio in M€ | Dipende dalla scala → non confrontabile tra indicatori diversi |
| **RMSE** | Come MAE, penalizza gli errori grandi | Stessa dipendenza dalla scala |
| **R²** | % variabilità spiegata (0–1) | Indipendente dalla scala → confrontabile tra modelli e indicatori |
| **R²_CV_media** | R² medio su 5 finestre temporali | Stima più stabile della vera capacità predittiva |

**R²_CV_media** è tecnicamente la stima più affidabile perché viene calcolata su 5 sotto-periodi
temporali diversi: se il modello è stabile in tutti i periodi, l'R²_CV sarà alto e a bassa varianza.
Un modello con R²_CV alto ma R²_test più basso potrebbe avere varianza elevata sugli anni finali.

---

### Il ruolo del Baseline

Il **Baseline** prevede semplicemente la mediana storica per ogni combinazione paese-settore,
senza alcun apprendimento automatico. Serve come **soglia minima**: un modello ML è utile
*solo se supera il Baseline*, altrimenti non aggiunge valore rispetto a una semplice statistica descrittiva.

Un R² Baseline molto alto (es. > 0.95) indica che la serie storica è molto regolare e prevedibile
anche senza ML — in questo caso il valore aggiunto dell'algoritmo è nella precisione e nella generalizzazione.

---

### Perché 4 modelli diversi?

- **Ridge** → modello lineare regolarizzato: robusto, interpretabile, ma limitato alle relazioni lineari
- **Random Forest** → ensemble di alberi: cattura relazioni non lineari, resistente agli outlier
- **Gradient Boosting** → apprendimento sequenziale degli errori: spesso il più preciso
- **MLP (Rete Neurale)** → percettrone multi-strato: potenzialmente molto espressivo, ma richiede più dati

Confrontarli permette di verificare se la complessità aggiuntiva (Gradient Boosting, MLP) porta
un reale beneficio rispetto a modelli più semplici (Ridge), principio noto come **parsimonia del modello**.
""")

        c1, c2 = st.columns([1.5, 1.5])
        with c1:
            st.markdown(f"#### Metriche – `{ind_sel}`")
            show_cols = [c for c in
                         ["Modello", "MAE", "RMSE", "R2",
                          "R2_CV_media", "R2_CV_std"]
                         if c in df_ci.columns]

            # Stile con contrasto alto: sfondo verde scuro + testo bianco grassetto
            _best_props  = "background-color: #1e8449; color: white; font-weight: bold;"
            _worst_props = "background-color: #c0392b; color: white; font-weight: bold;"

            _style = df_ci[show_cols].style
            _r2_cols  = [c for c in ["R2", "R2_CV_media"] if c in show_cols]
            _err_cols = [c for c in ["MAE", "RMSE"]       if c in show_cols]
            if _r2_cols:
                _style = _style.highlight_max(subset=_r2_cols, props=_best_props)
                _style = _style.highlight_min(subset=_r2_cols, props=_worst_props)
            if _err_cols:
                _style = _style.highlight_min(subset=_err_cols, props=_best_props)
                _style = _style.highlight_max(subset=_err_cols, props=_worst_props)
            _style = _style.format(
                {c: ("{:,.2f}" if c in ("MAE", "RMSE") else
                     ("{:.4f}"  if c == "R2"            else "{}"))
                 for c in show_cols if c != "Modello"}
            )
            st.dataframe(_style, use_container_width=True, hide_index=True)
            st.caption(
                "🟩 Verde scuro = valore migliore per quella metrica  |  "
                "🟥 Rosso = valore peggiore. "
                "R² e R²_CV_media: più alto = migliore. "
                "MAE e RMSE: più basso = migliore."
            )

            # Highlight modello vincente
            _df_ml_only = df_ci[df_ci["Modello"] != "Baseline (Mediana Storica)"]
            if len(_df_ml_only) and "R2" in _df_ml_only.columns:
                _best_row = _df_ml_only.nlargest(1, "R2").iloc[0]
                _bl_row   = df_ci[df_ci["Modello"] == "Baseline (Mediana Storica)"]
                _bl_r2    = float(_bl_row["R2"].iloc[0]) if len(_bl_row) else 0.0
                _guadagno = float(_best_row["R2"]) - _bl_r2
                st.success(
                    f"🏆 **Miglior modello per `{ind_sel}`:** **{_best_row['Modello']}**  "
                    f"— R² = **{float(_best_row['R2']):.4f}**  |  "
                    f"MAE = {float(_best_row['MAE']):,.2f} M€  |  "
                    f"Miglioramento vs Baseline: **+{_guadagno:.4f}** R²"
                )

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
                st.plotly_chart(fig_r2, use_container_width=True, key="pc_t3_r2bar")
                st.caption(
                    "📊 Confronto visivo dell'R² sul test set. "
                    "La barra più alta indica il modello che spiega meglio la variabilità dei dati. "
                    "Un R² negativo significa che il modello è peggiore di una semplice media storica."
                )

        st.markdown("---")
        st.markdown("#### R² su tutti gli indicatori – vista globale")
        st.caption(
            "ℹ️ **Questa sezione è indipendente da Paese e Settore** perché i modelli ML sono "
            "addestrati sull'intero dataset (tutti i paesi e settori insieme). "
            "Le metriche R², MAE e RMSE rappresentano la performance **globale** su tutto il test set. "
            f"L'indicatore attualmente selezionato nella sidebar è evidenziato: **{IND_LABELS.get(ind_sel,'').replace(' (M€)','')}**."
        )
        mn  = [k for k in all_res
               if k not in ("IND_LABELS", "cv", "baseline", "anno_cutoff")]
        r2d = [{"Modello": m, "Indicatore": i,
                "Descrizione": IND_LABELS.get(i, i),
                "R² Test": all_res[m][i].get("R2", 0)}
               for m in mn for i in INDICATORS]
        if "baseline" in all_res:
            r2d += [{"Modello": "Baseline", "Indicatore": i,
                     "Descrizione": IND_LABELS.get(i, i),
                     "R² Test": all_res["baseline"][i].get("R2", 0)}
                    for i in INDICATORS]
        _r2d_df = pd.DataFrame(r2d)
        _r2d_df["Indicatore Label"] = _r2d_df["Indicatore"].map(
            lambda x: IND_LABELS.get(x, x).replace(" (M€)", ""))
        # Etichetta dell'indicatore correntemente selezionato per evidenziarlo
        _ind_sel_label = IND_LABELS.get(ind_sel, ind_sel).replace(" (M€)", "")
        fig_all = px.bar(
            _r2d_df, x="Indicatore Label", y="R² Test",
            color="Modello", barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"Indicatore Label": "Indicatore"})
        # Rettangolo di evidenziazione sull'indicatore selezionato
        fig_all.add_vrect(
            x0=_ind_sel_label, x1=_ind_sel_label,
            fillcolor="rgba(255,255,255,0.08)",
            layer="below", line_width=0,
        )
        # Riga verticale tratteggiata per identificare l'indicatore corrente
        _x_vals = _r2d_df["Indicatore Label"].unique().tolist()
        if _ind_sel_label in _x_vals:
            fig_all.add_annotation(
                x=_ind_sel_label, y=1.08, xref="x", yref="paper",
                text="◀ selezionato", showarrow=False,
                font=dict(color="#f39c12", size=11), xanchor="center",
            )
        fig_all.update_layout(
            height=380, margin=dict(l=0, r=0, t=40, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_tickangle=-15,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_all, use_container_width=True, key="pc_t3_r2all")
        st.caption(
            "📊 Ogni gruppo di barre = R² dei 5 modelli per uno degli indicatori. "
            "Barre più alte = migliore generalizzazione. "
            "Se il Baseline (mediana storica) è già alto, il dataset è molto regolare. "
            "L'etichetta '◀ selezionato' indica l'indicatore corrente della sidebar."
        )

        # Riepilogo modelli vincenti (sempre globale, non cambia con paese/settore)
        _best_msgs = []
        for _i in INDICATORS:
            _best_ml = max(mn, key=lambda m: all_res[m][_i].get("R2", 0))
            _r2_ml   = all_res[_best_ml][_i].get("R2", 0)
            _r2_bl   = all_res["baseline"][_i].get("R2", 0) if "baseline" in all_res else 0
            _migl    = _r2_ml - _r2_bl
            _marker  = " ◀ **selezionato**" if _i == ind_sel else ""
            _best_msgs.append(
                f"**{IND_LABELS.get(_i, _i).replace(' (M€)','')}**{_marker}: "
                f"{_best_ml} — R²={_r2_ml:.3f} (+{_migl:.3f} vs baseline)"
            )
        with st.expander("🏆 Riepilogo modelli vincenti per indicatore", expanded=True):
            st.caption("Questa tabella non cambia al variare di Paese/Settore: è una proprietà globale dei modelli.")
            for _msg in _best_msgs:
                st.markdown(f"- {_msg}")

        # Feature Importance
        if fi_data:
            st.markdown("---")
            st.markdown(f"#### Feature Importance – `{ind_sel}`")
            st.caption(
                f"✅ **Questa sezione si aggiorna con l'indicatore** (attuale: `{ind_sel} – {IND_LABELS.get(ind_sel,'')}`). "
                "Non cambia con Paese o Settore: l'importanza è calcolata sull'intero dataset."
            )
            st.markdown(
                "Mostra il **peso percentuale** di ciascuna variabile nelle previsioni del modello. "
                "Settore NACE e Paese sono codificati con One-Hot Encoding: "
                "la barra mostra la somma dei pesi di tutte le categorie di quella variabile. "
                "Se il **Paese** domina: il modello ha imparato differenze strutturali tra nazioni. "
                "Se il **Settore** domina: le differenze tra settori economici sono il fattore principale. "
                "Se l'**Anno** domina: il trend temporale è il driver principale della variabile."
            )
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
                        st.plotly_chart(fig_fi, use_container_width=True,
                                        key=f"pc_t3_fi_{nm_fi.replace(' ','_')}")
                        st.caption(
                            f"📊 Importanza relativa delle 3 variabili per {nm_fi}. "
                            "La lunghezza della barra indica il peso percentuale di quella variabile "
                            "nel processo decisionale del modello. "
                            "Se 'Settore' domina, il modello ha imparato differenze strutturali tra settori; "
                            "se 'Anno' domina, il trend temporale è il fattore principale."
                        )


# ══ TAB 4 – ESPLORAZIONE ══════════════════════════════════════════════════════
with tab4:
    st.markdown("### Esplorazione Dataset Eurostat SBS")
    with st.expander("📖 Come esplorare il dataset", expanded=False):
        st.markdown("""
**Cosa fa questo tab?**
Offre una vista esplorativa del dataset Eurostat SBS (Structural Business Statistics) utilizzato
per addestrare i modelli ML. Permette di analizzare la distribuzione geografica e temporale
dei dati senza effettuare previsioni.

**Mappa geografica:**
Visualizza l'intensità dell'indicatore selezionato per ogni paese europeo in un anno specifico.
I colori più scuri indicano valori più alti. I paesi grigi non hanno dati disponibili per quell'anno.
Utile per confrontare la posizione dell'Italia rispetto agli altri paesi EU.

**Trend per settore:**
Mostra l'evoluzione temporale dell'indicatore per i settori NACE selezionati (2011–2020).
Permette di confrontare settori diversi sullo stesso grafico e identificare divergenze strutturali
o reazioni comuni a eventi economici (es. crisi 2008 o pandemia).

**Dati filtrati:**
Tabella grezza dei dati Eurostat per il paese e indicatore scelti nella sidebar.
Mostra i valori originali come forniti da Eurostat, prima di qualsiasi trasformazione ML.
""")
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
        st.plotly_chart(fig_map, use_container_width=True, key="pc_t4_map")
        st.caption(
            f"🗺️ Mappa coropletica: intensità di **{IND_LABELS[ind_sel]}** per paese europeo nell'anno {anno_m}. "
            "I colori più scuri = valori più alti. I paesi in grigio non hanno dati Eurostat disponibili. "
            "Passa il mouse su un paese per vedere il valore esatto."
        )

    with c2:
        st.markdown("#### Trend per settore")

        # Lista COMPLETA dei settori nel dataset (da tutti gli indicatori),
        # ordinata per valore medio decrescente dell'indicatore corrente.
        # In questo modo il multiselect mostra sempre tutti e 4 i settori,
        # indipendentemente dall'indicatore selezionato.
        _tutti_settori = sorted(df_full["nace_r2"].unique())
        _settori_con_dati = (df_ind.groupby("nace_r2")["OBS_VALUE"].mean()
                             .sort_values(ascending=False).index.tolist())
        # Settori con dati per l'indicatore corrente (per default e avvisi)
        _n_con_dati = len(_settori_con_dati)

        if _n_con_dati == 0:
            st.warning(f"Nessun dato nel dataset per **{IND_LABELS.get(ind_sel,'')}**.")
            ss = []
        else:
            if _n_con_dati < len(_tutti_settori):
                # V12130: avviso dati parziali con spiegazione metodologica
                _settori_mancanti = [s for s in _tutti_settori if s not in _settori_con_dati]
                st.warning(
                    f"⚠️ **{IND_LABELS.get(ind_sel,'')}** è disponibile nel dataset Eurostat SBS "
                    f"solo per **{_n_con_dati}** settore su {len(_tutti_settori)}: "
                    f"**{', '.join(nace_label(s) for s in _settori_con_dati)}**. "
                    f"I settori **{', '.join(nace_label(s) for s in _settori_mancanti)}** "
                    "non hanno osservazioni per questo indicatore nella fonte Eurostat."
                )
            else:
                st.caption(
                    f"**{_n_con_dati} settori** disponibili per {IND_LABELS.get(ind_sel,'')}. "
                    "Selezionati i primi 2 per valore medio — aggiungi o rimuovi liberamente."
                )

            # Multiselect: opzioni = settori CON dati per l'indicatore corrente
            # Default: i primi 2 per valore medio (lascia almeno 2 aggiungibili)
            _default_ts = _settori_con_dati[:min(2, _n_con_dati)]
            ss = st.multiselect(
                "Settori da confrontare",
                options=_settori_con_dati,
                default=_default_ts,
                format_func=nace_label,
                key="ts"
            )
        if ss:
            dt = (df_ind[df_ind["nace_r2"].isin(ss)]
                  .groupby(["TIME_PERIOD", "nace_r2"])["OBS_VALUE"]
                  .mean().reset_index())
            dt["Settore"] = dt["nace_r2"].map(nace_label)
            fig_tr = px.line(
                dt, x="TIME_PERIOD", y="OBS_VALUE",
                color="Settore", markers=True,
                labels={"TIME_PERIOD": "Anno",
                        "OBS_VALUE": IND_LABELS[ind_sel],
                        "Settore": "Settore"})
            fig_tr.update_layout(
                height=350, margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tr, use_container_width=True, key="pc_t4_trend")
            st.caption(
                f"📈 Evoluzione di **{IND_LABELS[ind_sel]}** per i settori selezionati. "
                "Ogni linea rappresenta un settore NACE. Clicca sulla legenda per mostrare/nascondere."
            )
            # Insight automatico: settore con maggior crescita/calo nel periodo
            _crescite_s = {}
            for _sec in ss:
                _ds = dt[dt["nace_r2"] == _sec].sort_values("TIME_PERIOD")
                if len(_ds) >= 2:
                    _v0s = float(_ds["OBS_VALUE"].iloc[0])
                    _v1s = float(_ds["OBS_VALUE"].iloc[-1])
                    if _v0s > 0:
                        _crescite_s[_sec] = (_v1s - _v0s) / _v0s * 100
            if len(_crescite_s) >= 2:
                _best_s  = max(_crescite_s, key=_crescite_s.get)
                _worst_s = min(_crescite_s, key=_crescite_s.get)
                _label_b = NACE_LABELS.get(_best_s, _best_s)
                _label_w = NACE_LABELS.get(_worst_s, _worst_s)
                st.info(
                    f"💡 **Trend nel periodo**: crescita maggiore → **{_label_b}** "
                    f"(**{_crescite_s[_best_s]:+.1f}%**) · "
                    f"calo maggiore → **{_label_w}** (**{_crescite_s[_worst_s]:+.1f}%**)"
                )
        else:
            st.info("Seleziona almeno un settore.")

    # ── Sezione strutturale: Numero di imprese (V11110) ───────────────────────
    st.markdown("---")
    st.markdown("#### 🏭 Numero di imprese attive (V11110)")
    st.caption(
        "Il numero di imprese è un **dato strutturale** che descrive la dimensione del mercato: "
        "quante PMI sono attive per settore e paese in ogni anno. "
        "Non è un indicatore di performance aziendale e non viene usato nei modelli ML, "
        "ma è utile per contestualizzare i benchmark e capire l'evoluzione del tessuto produttivo."
    )

    _df11 = df_full[df_full["indic_sb"] == IND_V11110]

    # Selezione settore e paesi da confrontare
    _col11s, _col11p = st.columns([1, 3])
    with _col11s:
        _settori11 = sorted(_df11["nace_r2"].unique())
        _settore11 = st.selectbox(
            "Settore", _settori11,
            index=_settori11.index(settore) if settore in _settori11 else 0,
            format_func=nace_label, key="s11")
    with _col11p:
        _paesi11_disp = sorted(_df11[_df11["nace_r2"] == _settore11]["geo"].unique())
        _default11 = [p for p in [paese, "IT", "DE", "FR"] if p in _paesi11_disp][:4]
        _paesi11_sel = st.multiselect(
            "Paesi da confrontare", _paesi11_disp,
            default=_default11,
            format_func=geo_label, key="p11")

    if _paesi11_sel:
        _dt11 = (_df11[
            (_df11["nace_r2"] == _settore11) &
            (_df11["geo"].isin(_paesi11_sel))
        ].sort_values("TIME_PERIOD").copy())
        _dt11["Paese"] = _dt11["geo"].map(geo_label)

        _fig11 = px.line(
            _dt11, x="TIME_PERIOD", y="OBS_VALUE",
            color="Paese", markers=True,
            labels={"TIME_PERIOD": "Anno", "OBS_VALUE": "N° imprese", "Paese": "Paese"},
            title=f"Andamento numero di PMI – {nace_label(_settore11)}")
        _fig11.update_layout(
            height=380, margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(_fig11, use_container_width=True, key="pc_t4_v11110")
        st.caption(
            f"📈 Ogni linea rappresenta un paese selezionato. "
            f"Settore: **{nace_label(_settore11)}** — fonte Eurostat SBS (V11110). "
            "Un calo progressivo indica contrazione del tessuto imprenditoriale; "
            "una crescita indica sviluppo del settore nel paese."
        )
    else:
        st.info("Seleziona almeno un paese dal menu sopra.")

    st.markdown("---")
    st.markdown("#### Dati grezzi Eurostat")
    st.caption(
        f"Valori originali Eurostat SBS per **{geo_label(paese)}** – indicatore **{ind_sel}** "
        "(in milioni di euro). Sono i dati di input usati per addestrare i modelli ML, "
        "prima di qualsiasi trasformazione."
    )
    df_raw = (df_full[(df_full["indic_sb"] == ind_sel) & (df_full["geo"] == paese)]
              .sort_values("TIME_PERIOD", ascending=False))
    st.dataframe(df_raw.reset_index(drop=True).head(50), use_container_width=True)
    st.caption(f"Visualizzate max 50 righe su {len(df_raw)} disponibili per {geo_label(paese)} – {ind_sel}")


# ══ TAB 5 – VALUTAZIONE AZIENDALE ════════════════════════════════════════════
with tab5:
    import plotly.graph_objects as go

    st.markdown("### 🏢 Valutazione Aziendale")
    st.markdown(
        "Inserisci i dati della tua azienda per confrontarli con il **benchmark "
        "ML** del settore di riferimento. Il sistema calcola lo scostamento, "
        "il z-score rispetto alla media storica e classifica la performance.")

    # ── Nota metodologica ──────────────────────────────────────────────────────
    with st.expander("📖 Guida alla valutazione aziendale – come leggere i risultati", expanded=False):
        st.markdown("""
### Come funziona la valutazione?

Inserisci i dati della tua azienda (paese, settore, anno, indicatore e valore) e il sistema
confronta automaticamente il tuo valore con il **benchmark ML** del settore di riferimento.

---

### I 4 valori che vedi nei risultati

**📌 Benchmark per PMI (M€)**
Il modello ML (Ridge / Random Forest / Gradient Boosting / MLP) prevede il valore *aggregato* per
l'intero settore in quel paese e anno. Questo aggregato viene poi diviso per il numero di imprese
attive nel settore (V11110, fonte Eurostat) per ottenere il valore atteso *per singola PMI*.
Questo è il punto di riferimento con cui viene confrontato il tuo dato.

**📌 Gap vs benchmark**
È la differenza tra il tuo valore e il benchmark per PMI.
- Valore positivo = la tua azienda è **sopra** il benchmark (🟢 per ricavi/VA, 🔴 per costi)
- Valore negativo = la tua azienda è **sotto** il benchmark (🔴 per ricavi/VA, 🟢 per costi)
- La percentuale indica l'entità dello scostamento rispetto al benchmark

**📌 Z-score**
Misura quanto il tuo valore si discosta dalla *media storica del settore* in unità di deviazione standard (σ).
- **z > +0.5σ** → performance sopra la media del settore
- **z tra −0.5 e +0.5σ** → performance nella norma
- **z < −0.5σ** → performance sotto la media del settore
Il vantaggio dello z-score è che è robusto rispetto ai valori estremi e funziona anche
quando la distribuzione è asimmetrica.

**📌 Valutazione finale**
Combina z-score e gap percentuale per assegnare una fascia:
- 🟢 **Alta Performance / Efficienza Costi** – valore significativamente sopra (ricavi) o sotto (costi) la media
- 🟡 **Media Performance / Costi Allineati** – valore nella norma del settore
- 🔴 **Bassa Performance / Costi Elevati** – valore significativamente sotto (ricavi) o sopra (costi) la media

---

### I due grafici

**Gauge (manometro):** mostra lo z-score su una scala da −3σ a +3σ.
La zona verde indica la direzione desiderabile (destra per ricavi, sinistra per costi).

**Violin plot:** mostra la distribuzione storica di tutte le PMI europee nel settore.
La linea rossa tratteggiata è il tuo valore; la linea verde è il benchmark ML.
Più la tua linea è in alto nel grafico (per ricavi) o in basso (per costi), meglio è.

---

> *Nota: tutti i valori del dataset Eurostat SBS sono **aggregati settoriali** (in M€).
> Il confronto con la singola PMI è possibile grazie alla normalizzazione per numero di imprese (V11110).*
""")

    st.markdown("---")

    # ── Form di input ──────────────────────────────────────────────────────────
    st.markdown("#### Dati azienda da valutare")
    st.caption(
        "Seleziona paese, settore e anno di riferimento, poi inserisci il valore del tuo indicatore. "
        "Il sistema confronterà il dato con la media storica del settore calcolata dal modello ML."
    )

    # Settori e paesi disponibili nel dataset (non tutti i codici NACE)
    _SETTORI_DATI = sorted(df_full["nace_r2"].unique())
    _PAESI_DATI   = sorted(df_full["geo"].unique())
    _EU_AGGREGATI = {"EU27_2020", "EU28"}
    # V12130 presente solo nel dataset per settore B
    _V12130_SETTORI_CON_DATI = sorted(
        df_full[df_full["indic_sb"] == "V12130"]["nace_r2"].unique())

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        val_paese = st.selectbox(
            "🌍 Paese",
            _PAESI_DATI,
            index=_PAESI_DATI.index("IT") if "IT" in _PAESI_DATI else 0,
            format_func=geo_label,
            key="val_paese",
        )
        if val_paese in _EU_AGGREGATI:
            st.caption(
                f"⚠️ **{val_paese}** è un aggregato europeo (non un singolo paese). "
                "Il benchmark per-PMI sarà la media su tutte le PMI dell'UE, "
                "non quella nazionale.")
    with fc2:
        val_settore = st.selectbox(
            "🏭 Settore NACE",
            _SETTORI_DATI,
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

    # V11110 = conteggio settoriale, non metrica di performance: escluso dalla valutazione
    _INDICATORI_PERF = [i for i in INDICATORS if INDICATOR_ORIENTATION.get(i) != "count"]

    fv1, fv2 = st.columns(2)
    with fv1:
        val_indicatore = st.selectbox(
            "📌 Indicatore da analizzare",
            _INDICATORI_PERF,
            format_func=lambda x: f"{x} – {IND_LABELS[x]}",
            key="val_ind",
        )
        st.caption(f"ℹ️ {IND_DESCRIPTIONS[val_indicatore]}")
        # Avviso: V12130 presente solo per alcuni settori nel dataset
        if val_indicatore == "V12130" and val_settore not in _V12130_SETTORI_CON_DATI:
            st.warning(f"⚠️ V12130 non disponibile per {nace_label(val_settore)}: benchmark stimato per estrapolazione.")
        # Avviso: V12130 può avere benchmark negativi nel settore B
        if val_indicatore == "V12130" and val_settore == "B":
            st.caption(f"ℹ️ Il Valore aggiunto per {nace_label(val_settore)} può essere negativo in alcuni paesi/anni.")
    with fv2:
        # Carica statistiche di settore per range e default realistici
        _rng = stats_settore_indicatore(val_indicatore, val_settore)
        _default_val = float(round(_rng["mediana"], 2)) if _rng else 1000.0
        _step_val    = max(1.0, float(round(_rng["std"] / 100, 1))) if _rng else 10.0

        _orient_rng = INDICATOR_ORIENTATION.get(val_indicatore, "positive")
        val_reale = st.number_input(
            f"💰 Valore reale azienda  ({IND_LABELS.get(val_indicatore, '')})",
            min_value=0.0,
            value=_default_val,
            step=_step_val,
            format="%.2f",
            key="val_reale",
        )

        # ── Hint rapido: range di valori realistici ────────────────────────────
        if _rng and _orient_rng != "count":
            _has_pe_hint = "pe_mediana" in _rng
            if _has_pe_hint:
                _hint_lo  = max(0.0, _rng["pe_p5"])
                _hint_hi  = _rng["pe_p95"]
                _hint_med = _rng["pe_mediana"]
                _hint_src = "per singola PMI europea"
            else:
                _hint_lo  = max(0.0, _rng["p5"])
                _hint_hi  = _rng["p95"]
                _hint_med = _rng["mediana"]
                _hint_src = "aggregato settoriale europeo"
            _cost_note = " · valori più bassi = maggiore efficienza" if _orient_rng == "cost" else ""
            # Controlla se esiste un dato storico per il paese selezionato
            _df_paese_hint = df_full[
                (df_full["indic_sb"] == val_indicatore) &
                (df_full["nace_r2"]  == val_settore) &
                (df_full["geo"]      == val_paese)
            ]["OBS_VALUE"].dropna()
            _paese_note = ""
            # n_ent_v è disponibile solo dopo il click del bottone;
            # qui calcoliamo localmente la media imprese per il paese selezionato
            _n_ent_hint_vals = df_full[
                (df_full["indic_sb"] == "V11110") &
                (df_full["nace_r2"]  == val_settore) &
                (df_full["geo"]      == val_paese)
            ]["OBS_VALUE"].dropna()
            _n_ent_hint = (float(_n_ent_hint_vals.mean())
                           if len(_n_ent_hint_vals) > 0 and _n_ent_hint_vals.mean() > 0
                           else None)
            if len(_df_paese_hint) > 0 and _n_ent_hint:
                _med_paese_pe = float(_df_paese_hint.mean()) / _n_ent_hint
                _paese_note = (
                    f" · Storico **{GEO_LABELS.get(val_paese, val_paese)}** "
                    f"(media per PMI): **{_med_paese_pe:,.2f} M€**"
                )
            st.caption(
                f"📝 Range europeo ({_hint_src}): "
                f"tra **{_hint_lo:,.2f}** e **{_hint_hi:,.2f}** M€ "
                f"(P5–P95, mediana {_hint_med:,.2f} M€)"
                f"{_cost_note}{_paese_note}"
            )


    st.markdown("")
    btn_valuta = st.button("▶ Valuta l'azienda", type="primary",
                           use_container_width=False)

    if btn_valuta:
        _orient_v = INDICATOR_ORIENTATION.get(val_indicatore, "positive")

        # ── Carica modelli ─────────────────────────────────────────────────────
        pay_v = carica_modello(val_indicatore)
        kd_v  = carica_kmeans(val_indicatore)

        if pay_v is None:
            st.error("Modello ML non trovato. Avvia prima il training dalla sidebar.")
        else:
            # ── Previsione benchmark ML (aggregato di settore) ─────────────────
            Xnew = pd.DataFrame({
                "nace_r2":     [val_settore],
                "geo":         [val_paese],
                "TIME_PERIOD": [val_anno],
            })
            try:
                benchmark_agg = float(pay_v["model"].predict(Xnew)[0])
            except Exception as ex:
                st.error(f"Errore nella previsione: {ex}")
                st.stop()

            # ── Normalizzazione per-impresa (aggregato ÷ V11110) ───────────────
            n_ent_v = None
            try:
                r_n = df_full[
                    (df_full["indic_sb"] == "V11110") &
                    (df_full["nace_r2"]  == val_settore) &
                    (df_full["geo"]      == val_paese) &
                    (df_full["TIME_PERIOD"] == val_anno)
                ]["OBS_VALUE"].dropna()
                if len(r_n) > 0 and float(r_n.iloc[0]) > 0:
                    n_ent_v = float(r_n.iloc[0])
                else:
                    # Fallback: media degli anni disponibili
                    r_n2 = df_full[
                        (df_full["indic_sb"] == "V11110") &
                        (df_full["nace_r2"]  == val_settore) &
                        (df_full["geo"]      == val_paese)
                    ]["OBS_VALUE"].dropna()
                    if len(r_n2) > 0 and r_n2.mean() > 0:
                        n_ent_v = float(r_n2.mean())
            except Exception:
                pass

            benchmark_pe = (benchmark_agg / n_ent_v
                            if n_ent_v and n_ent_v > 0 else None)

            # Usa per-impresa come benchmark primario; aggregato come secondario
            benchmark = benchmark_pe if benchmark_pe is not None else benchmark_agg
            gap_abs   = val_reale - benchmark
            gap_pct   = (gap_abs / benchmark * 100) if benchmark != 0 else 0.0

            # ── Gestione benchmark negativo ────────────────────────────────────
            # Due casi distinti:
            # A) V12130 (Valore Aggiunto) settore B: un VA negativo è economicamente
            #    possibile (costi > ricavi). Si mantiene il benchmark e si usa solo z-score.
            # B) V12110/V12120/V12150 (Fatturato, Produzione, Personale): un valore
            #    negativo è fisicamente impossibile → il modello ha estrapolato fuori
            #    dal dominio (pochi dati per quel paese-settore-anno).
            #    Fallback: mediana storica europea del settore.
            _bench_negativo_originale = benchmark < 0
            _benchmark_agg_raw       = benchmark_agg  # conserva per il messaggio

            # Indicatori per cui un negativo è impossibile economicamente
            _IND_NON_NEGATIVI = {"V12110", "V12120", "V12150"}
            _bench_impossibile = (
                _bench_negativo_originale and
                val_indicatore in _IND_NON_NEGATIVI
            )

            if _bench_impossibile:
                # Fallback: mediana storica per-impresa del settore (dati europei)
                _rng_fb = stats_settore_indicatore(val_indicatore, val_settore)
                if _rng_fb and "pe_mediana" in _rng_fb and n_ent_v and n_ent_v > 0:
                    benchmark_pe  = float(_rng_fb["pe_mediana"])
                    benchmark_agg = benchmark_pe * n_ent_v
                elif _rng_fb and "mediana" in _rng_fb:
                    benchmark_agg = float(_rng_fb["mediana"])
                    benchmark_pe  = (benchmark_agg / n_ent_v
                                     if n_ent_v and n_ent_v > 0 else None)
                # Ricalcola tutto con il fallback
                benchmark = benchmark_pe if benchmark_pe is not None else benchmark_agg
                gap_abs   = val_reale - benchmark
                gap_pct   = (gap_abs / benchmark * 100) if benchmark != 0 else 0.0

            # Stato finale del benchmark dopo eventuale fallback
            _bench_negativo = benchmark < 0  # True solo se anche fallback è negativo (VA)
            _gp_class = 0.0 if _bench_negativo else gap_pct

            # ── Statistiche distribuzione settore (calcolate una volta sola) ──────
            # Usate per: flag P95, violin, Contesto di mercato, interpretazione
            _rng_settore = stats_settore_indicatore(val_indicatore, val_settore)
            # Flag: benchmark ML supera il 95° percentile storico per-PMI
            # → il modello ha probabilmente estrapolato fuori dal range normale
            _bench_sopra_p95 = (
                _rng_settore is not None and
                "pe_p95" in _rng_settore and
                benchmark_pe is not None and
                benchmark_pe > _rng_settore["pe_p95"]
            )

            # ── Z-score (su valori per-impresa se disponibili) ─────────────────
            zscore_val  = None
            profilo_val = "N/D"
            mu_val      = None
            sigma_val   = None

            if kd_v is not None:
                try:
                    if "sector_stats" in kd_v and val_settore in kd_v["sector_stats"].index:
                        mu_agg    = float(kd_v["sector_stats"].loc[val_settore, "mu"])
                        sig_agg   = float(kd_v["sector_stats"].loc[val_settore, "sigma"])
                    else:
                        mu_agg  = float(df_full[df_full["indic_sb"] == val_indicatore
                                                ]["OBS_VALUE"].median())
                        sig_agg = 1.0
                    sig_agg = max(sig_agg, 1e-6)

                    # Normalizza mu/sigma alla scala per-impresa
                    if n_ent_v and n_ent_v > 0:
                        mu_val    = mu_agg  / n_ent_v
                        sigma_val = sig_agg / n_ent_v
                    else:
                        mu_val    = mu_agg
                        sigma_val = sig_agg
                    sigma_val = max(sigma_val, 1e-9)

                    zscore_val  = float(np.clip(
                        (val_reale - mu_val) / sigma_val, -3, 3))
                    Xcl_v = kd_v["scaler"].transform(
                        [[float(np.clip((val_reale * (n_ent_v or 1) - mu_agg) / sig_agg, -3, 3)),
                          val_anno]])
                    lbl_v = int(kd_v["km"].predict(Xcl_v)[0])
                    profilo_val = kd_v["label_map"].get(lbl_v, "N/D")
                except Exception:
                    pass

            # ── Determina fascia interpretativa ───────────────────────────────
            # Principi:
            #  • Indicatori positivi (V12110/20/30): valori alti = meglio
            #  • Indicatori di costo  (V12150):      valori bassi = meglio
            #  • Se benchmark < 0: gap_pct è matematicamente invertito →
            #    si usa SOLO lo z-score (già corretto per distribuzioni asimmetriche)
            #  • Safety net: gap_pct estremo (±50% / ±100%) integra lo z-score
            #    per distribuzioni ad alta varianza (σ >> μ)
            _z  = zscore_val if zscore_val is not None else 0.0
            _gp = _gp_class   # 0 se benchmark negativo, altrimenti gap_pct reale

            if _orient_v == "cost":
                if _z <= -0.5 or _gp <= -50:
                    fascia_interp = "Efficienza Costi"
                elif _z >= 0.5 or _gp >= 50:
                    fascia_interp = "Costi Elevati"
                else:
                    fascia_interp = "Costi Allineati"
            else:
                _z_alta  = (zscore_val is not None and _z >= 0.5) or _gp >= 100
                _z_bassa = (zscore_val is not None and _z <= -0.5) or _gp <= -50
                _segnali_conflitto = False
                if _z_alta and not _z_bassa:
                    fascia_interp = "Alta Performance"
                elif _z_bassa and not _z_alta:
                    fascia_interp = "Bassa Performance"
                elif _z_alta and _z_bassa:
                    # Caso conflitto: z-score ≥ 0.5 (azienda sopra media europea)
                    # MA gap% ≤ -50% (sotto il benchmark ML specifico per paese).
                    # Accade quando il modello prevede un benchmark insolitamente alto
                    # per quel paese (es. extrapolazione fuori range storico).
                    # Classificazione conservativa: Media Performance.
                    fascia_interp = "Media Performance"
                    _segnali_conflitto = True
                else:
                    fascia_interp = "Media Performance"

            # ── Nota metodologica ──────────────────────────────────────────────
            st.markdown("---")

            # ── Avvisi benchmark: testo differenziato per indicatore ─────────────
            if _bench_impossibile:
                # Modello ha estrapolato → valore negativo impossibile → fallback usato
                st.warning(
                    f"⚠️ **Stima ML non affidabile** per questa combinazione: "
                    f"il modello ha prodotto un valore aggregato negativo "
                    f"({_benchmark_agg_raw:,.2f} M€) per "
                    f"**{IND_LABELS[val_indicatore]}** "
                    f"in {geo_label(val_paese)} – {nace_label(val_settore)} ({val_anno}). "
                    "Per questo indicatore un valore negativo non ha senso economico "
                    "(il modello ha troppi pochi dati storici per questo paese-settore). "
                    "**Il confronto è stato eseguito usando la mediana storica europea "
                    "del settore come benchmark alternativo.**"
                )
            elif _bench_negativo:
                # Solo V12130 – Valore Aggiunto: negativo economicamente possibile
                st.warning(
                    f"⚠️ **Benchmark ML negativo** ({benchmark:,.4f} M€): "
                    f"il modello prevede un Valore Aggiunto negativo per "
                    f"{geo_label(val_paese)} – {nace_label(val_settore)} ({val_anno}). "
                    "Questo è economicamente possibile nel settore estrattivo "
                    "quando i costi operativi superano i ricavi (VA < 0). "
                    "Il gap% non è applicabile: la valutazione si basa "
                    "**esclusivamente sullo z-score**."
                )

            # ── Avviso benchmark sopra P95 (extrapolazione del modello) ─────────
            if _bench_sopra_p95 and not _bench_impossibile:
                st.warning(
                    f"⚠️ **Benchmark ML elevato**: il valore previsto per singola PMI "
                    f"(**{benchmark_pe:,.2f} M€/PMI**) supera il 95° percentile storico "
                    f"europeo del settore "
                    f"(**{_rng_settore['pe_p95']:,.2f} M€/PMI**). "
                    "Il modello sta estrapolando fuori dal range normale per questa "
                    "combinazione paese-settore-anno (dati storici limitati o struttura "
                    "del settore molto diversa dalla media europea). "
                    "Usa il range P5–P95 del riquadro di inserimento come riferimento "
                    "alternativo più robusto."
                )

            if benchmark_pe is not None:
                st.caption(
                    f"📐 Benchmark per singola PMI: **{benchmark_pe:,.4f} M€** "
                    f"(aggregato ML {benchmark_agg:,.2f} M€ ÷ {n_ent_v:,.0f} imprese)"
                )
            else:
                st.caption(
                    f"⚠️ N° imprese non disponibile – confronto sull'aggregato di settore ({benchmark_agg:,.2f} M€)."
                )

            st.markdown("#### Risultati della valutazione")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Benchmark per PMI (M€)",
                f"{benchmark:,.4f}" if benchmark < 100 else f"{benchmark:,.2f}",
                help=(f"Valore atteso dal modello ML per una singola PMI nel settore. "
                      f"Calcolato come: aggregato ML {benchmark_agg:,.0f} M€ ÷ {n_ent_v:,.0f} imprese attive nel settore."
                      if benchmark_pe else
                      f"Valore aggregato previsto dal modello ML per l'intero settore: {benchmark_agg:,.2f} M€. "
                      "Non è stato possibile normalizzare per singola impresa (V11110 mancante)."),
            )
            k2.metric(
                "Valore inserito (M€)",
                f"{val_reale:,.4f}" if val_reale < 100 else f"{val_reale:,.2f}",
                help="Il valore che hai inserito per la tua azienda, espresso in milioni di euro (M€). "
                     "Viene confrontato con il benchmark ML per calcolare il gap e lo z-score.",
            )
            _delta_col = ("normal" if _orient_v == "positive"
                          else "inverse" if _orient_v == "cost"
                          else "off")
            # Se benchmark negativo il gap% è matematicamente invertito: mostriamo
            # solo il gap assoluto e una nota, senza la percentuale fuorviante
            if _bench_negativo:
                k3.metric(
                    "Gap assoluto (M€)",
                    f"{gap_abs:+,.4f}" if abs(gap_abs) < 100 else f"{gap_abs:+,.2f}",
                    help="Differenza assoluta tra il tuo valore e il benchmark ML. "
                         "Gap% non mostrato: con benchmark negativo il rapporto percentuale "
                         "è matematicamente invertito e non interpretabile.",
                )
            else:
                k3.metric(
                    "Gap vs benchmark",
                    f"{gap_abs:+,.4f}" if abs(gap_abs) < 100 else f"{gap_abs:+,.2f}",
                    delta=f"{gap_pct:+.1f}%",
                    delta_color=_delta_col,
                    help="Differenza assoluta (M€) e percentuale (%) tra il tuo valore e il benchmark ML. "
                         "Per indicatori di ricavo/VA: verde = sopra benchmark. "
                         "Per costi (V12150): verde = sotto benchmark (efficienza maggiore).",
                )
            if zscore_val is not None:
                k4.metric(
                    "Z-score (scala per-PMI)",
                    f"{zscore_val:+.2f} σ",
                    help="Scostamento dalla media storica del settore, espresso in deviazioni standard (σ). "
                         "z > +0.5σ: sopra la media. z < −0.5σ: sotto la media. "
                         "Per i costi l'interpretazione è invertita: z negativo = costi più bassi = efficienza.",
                )

            # ── Badge profilo ──────────────────────────────────────────────────
            _css_fascia = {
                "Alta Performance":  "risk-alta",
                "Media Performance": "risk-media",
                "Bassa Performance": "risk-bassa",
                "Efficienza Costi":  "risk-alta",   # costi bassi = verde
                "Costi Allineati":   "risk-media",
                "Costi Elevati":     "risk-bassa",  # costi alti = rosso
            }
            css_badge = _css_fascia.get(fascia_interp, "risk-media")
            st.markdown(
                f"**Valutazione:**&nbsp;&nbsp;"
                f'<span class="{css_badge}">{fascia_interp}</span>'
                + (f"&nbsp;&nbsp;<span style='color:#aaa;font-size:0.85em;'>"
                   f"(K-Means: {profilo_val})</span>" if profilo_val != "N/D" else ""),
                unsafe_allow_html=True,
            )
            st.markdown("")

            # ── Grafici: gauge + violin ────────────────────────────────────────
            col_g, col_p = st.columns([1.1, 1.9])

            with col_g:
                zv = round(zscore_val, 4) if zscore_val is not None else 0.0
                # Per costi: i colori delle zone sono INVERTITI
                if _orient_v == "cost":
                    _zone_lo = "#eafaf1"   # verde a sinistra (costi bassi = buono)
                    _zone_hi = "#fde8e8"   # rosso a destra (costi alti = brutto)
                    _bar_col = ("#27ae60" if zv <= -0.5
                                else "#e74c3c" if zv >= 0.5
                                else "#f39c12")
                    _gauge_title = "Z-score costi (–=efficiente, +=oneroso)"
                else:
                    _zone_lo = "#fde8e8"
                    _zone_hi = "#eafaf1"
                    _bar_col = ("#27ae60" if zv >= 0.5
                                else "#e74c3c" if zv <= -0.5
                                else "#f39c12")
                    _gauge_title = "Z-score performance (+= sopra media)"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=zv,
                    delta={"reference": 0, "suffix": " σ", "valueformat": "+.2f"},
                    number={"suffix": " σ", "valueformat": "+.2f"},
                    gauge={
                        "axis": {"range": [-3, 3], "tickwidth": 1,
                                 "tickcolor": "#555",
                                 "tickvals": [-3, -2, -1, 0, 1, 2, 3]},
                        "bar":  {"color": _bar_col},
                        "bgcolor": "white",
                        "steps": [
                            {"range": [-3,  -0.5], "color": _zone_lo},
                            {"range": [-0.5, 0.5], "color": "#fef9e7"},
                            {"range": [ 0.5,  3],  "color": _zone_hi},
                        ],
                        "threshold": {
                            "line": {"color": "#2c3e50", "width": 3},
                            "thickness": 0.75,
                            "value": zv,
                        },
                    },
                    title={"text": _gauge_title, "font": {"size": 12}},
                ))
                fig_gauge.update_layout(
                    height=280, margin=dict(l=20, r=20, t=60, b=10),
                    paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True, key="pc_t5_gauge")
                if _orient_v == "cost":
                    st.caption(
                        "🎯 **Gauge Z-score** per indicatore di costo. "
                        "La zona **verde a sinistra** (z negativo) indica costi sotto la media = efficienza. "
                        "La zona **rossa a destra** (z positivo) indica costi sopra la media. "
                        "L'ago punta al valore z-score della tua azienda rispetto al settore."
                    )
                else:
                    st.caption(
                        "🎯 **Gauge Z-score** per indicatore di ricavo/VA. "
                        "La zona **verde a destra** (z positivo) indica performance sopra la media. "
                        "La zona **rossa a sinistra** (z negativo) indica performance sotto la media. "
                        "L'ago punta al valore z-score della tua azienda rispetto al settore."
                    )

            with col_p:
                # Distribuzione per-impresa (se disponibile), altrimenti aggregato
                # Riusa _rng_settore già calcolato in precedenza (no chiamata doppia)
                _rng_v = _rng_settore
                _use_pe = (_rng_v is not None and "pe_mediana" in _rng_v
                           and n_ent_v and n_ent_v > 0)

                if _use_pe:
                    # Ricostruisci serie per-impresa dal dataset
                    _df_v = df_full[
                        (df_full["indic_sb"] == val_indicatore) &
                        (df_full["nace_r2"]  == val_settore)
                    ].merge(
                        df_full[(df_full["indic_sb"] == "V11110") &
                                (df_full["nace_r2"]  == val_settore)]
                        [["geo", "TIME_PERIOD", "OBS_VALUE"]]
                        .rename(columns={"OBS_VALUE": "n_ent"}),
                        on=["geo", "TIME_PERIOD"], how="left"
                    )
                    _df_v["n_ent"] = pd.to_numeric(
                        _df_v["n_ent"], errors="coerce").replace(0, np.nan)
                    _df_v["val_pe"] = _df_v["OBS_VALUE"] / _df_v["n_ent"]
                    _dist_pe = _df_v["val_pe"].dropna()
                    _violin_y = _dist_pe
                    _violin_title = f"Distribuzione per-PMI – {nace_label(val_settore)}"
                else:
                    _violin_y = df_full[
                        (df_full["indic_sb"] == val_indicatore) &
                        (df_full["nace_r2"]  == val_settore)
                    ]["OBS_VALUE"].dropna()
                    _violin_title = f"Distribuzione aggregato – {nace_label(val_settore)}"

                if len(_violin_y) >= 5:
                    st.markdown(f"**{_violin_title}**")
                    fig_pos = go.Figure()
                    fig_pos.add_trace(go.Violin(
                        y=_violin_y,
                        name="Distribuzione", box_visible=True,
                        meanline_visible=True, fillcolor="#aed6f1",
                        opacity=0.6, line_color="#1a5276", points="outliers",
                    ))
                    # Posiziona le annotazioni in lati opposti se i valori sono vicini
                    _ann_range = float(_violin_y.max() - _violin_y.min()) if len(_violin_y) else 1.0
                    _ann_range = max(_ann_range, 1e-9)
                    _valori_vicini = abs(val_reale - benchmark) / _ann_range < 0.05
                    _pos_azi   = "top left"    if _valori_vicini else "top right"
                    _pos_bench = "bottom right"
                    fig_pos.add_hline(
                        y=val_reale, line_dash="dash", line_color="#e74c3c",
                        line_width=2,
                        annotation_text=f"Tua azienda: {val_reale:,.4f}",
                        annotation_position=_pos_azi,
                        annotation_font_color="#e74c3c",
                    )
                    fig_pos.add_hline(
                        y=benchmark, line_dash="dot", line_color="#27ae60",
                        line_width=2,
                        annotation_text=f"Benchmark ML: {benchmark:,.4f}",
                        annotation_position=_pos_bench,
                        annotation_font_color="#27ae60",
                    )
                    fig_pos.update_layout(
                        height=280, margin=dict(l=0, r=0, t=10, b=0),
                        yaxis_title=IND_LABELS[val_indicatore],
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_pos, use_container_width=True, key="pc_t5_violin")
                    st.caption(
                        "🎻 **Violin plot**: distribuzione storica dei valori per-PMI nel settore "
                        f"**{nace_label(val_settore)}** (tutti i paesi, 2011–2020). "
                        "La forma del violino mostra dove si concentrano i valori — più largo = più osservazioni. "
                        "La **linea rossa tratteggiata** è il valore della tua azienda; "
                        "la **linea verde punteggiata** è il benchmark ML. "
                        + ("Più in basso = minori costi = meglio." if _orient_v == "cost"
                           else "Più in alto = performance maggiore = meglio.")
                    )
                else:
                    st.info("Dati insufficienti per il grafico di distribuzione.")

            # ── Riepilogo sintetico ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Riepilogo")
            righe_riepilogo = {
                "Paese / Settore / Anno":   f"{geo_label(val_paese)}  ·  {nace_label(val_settore)}  ·  {val_anno}",
                "Indicatore":               IND_LABELS[val_indicatore],
                "Valore inserito (M€)":     f"{val_reale:,.4f}",
                "Benchmark per PMI (M€)":   (f"{benchmark:,.4f}" if benchmark_pe
                                             else f"{benchmark_agg:,.2f} (aggregato)"),
                "Gap vs benchmark":         (f"{gap_abs:+,.4f} M€  ({gap_pct:+.1f}%)"
                                             if not _bench_negativo
                                             else f"{gap_abs:+,.4f} M€"),
                "Z-score":                  (f"{zscore_val:+.2f} σ"
                                             if zscore_val is not None else "N/D"),
                "Valutazione":              fascia_interp,
            }
            st.dataframe(
                pd.DataFrame(list(righe_riepilogo.items()),
                             columns=["Parametro", "Valore"]),
                use_container_width=True, hide_index=True)

            # ── Interpretazione ed Ipotesi ────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💡 Interpretazione ed Ipotesi")
            _zstr = (f" (z-score: **{zscore_val:+.2f}σ**)"
                     if zscore_val is not None else "")

            if _orient_v == "cost":
                # V12150 – Spese per il personale: logica INVERTITA
                if fascia_interp == "Efficienza Costi":
                    st.success(
                        f"✅ **Buona efficienza del costo del lavoro**: le spese per il "
                        f"personale sono **{gap_pct:+.1f}%** al di sotto del benchmark "
                        f"per singola PMI{_zstr}. Un costo del lavoro inferiore alla "
                        "media indica un'organizzazione efficiente. Da verificare che la "
                        "riduzione dei costi non comprometta capacità produttiva o qualità."
                    )
                elif fascia_interp == "Costi Elevati":
                    st.error(
                        f"⚠️ **Costi del personale elevati**: le spese per il personale "
                        f"sono **{abs(gap_pct):.1f}%** superiori al benchmark per singola "
                        f"PMI{_zstr}. Costi del lavoro significativamente sopra la media "
                        "del settore possono indicare: forza lavoro più numerosa, retribuzioni "
                        "più alte o bassa produttività per addetto. Si raccomanda l'analisi "
                        "del rapporto Costo lavoro / Fatturato (V12150 / V12110) e il "
                        "confronto con le best practice del settore."
                    )
                else:
                    st.info(
                        f"ℹ️ **Costi del personale in linea**: gap **{gap_pct:+.1f}%**"
                        f"{_zstr}. Le spese per il personale sono allineate alla media "
                        "del settore. Per ottimizzare l'efficienza, analizzare il rapporto "
                        "V12150 / V12110 (Costo lavoro su Fatturato) rispetto ai competitor."
                    )
            else:
                # Indicatori positivi: V12110, V12120, V12130
                if fascia_interp == "Alta Performance":
                    st.success(
                        f"✅ L'azienda **supera il benchmark per-PMI** del **{gap_pct:+.1f}%**"
                        f"{_zstr}, posizionandosi in una zona di **alta performance** "
                        f"rispetto alla media storica del settore **{nace_label(val_settore)}**. "
                        "Questo suggerisce un vantaggio competitivo o un ciclo di crescita sostenuta."
                    )
                elif fascia_interp == "Media Performance":
                    if _segnali_conflitto:
                        # Caso speciale: z-score positivo (sopra media EU) ma gap% negativo
                        # (sotto il benchmark ML paese-specifico, che è anormalmente alto)
                        _p95_str = (f" – P95 europeo: **{_rng_settore['pe_p95']:,.2f} M€/PMI**"
                                    if _rng_settore and "pe_p95" in _rng_settore else "")
                        st.warning(
                            f"📊 **Segnali in contrasto**: lo z-score (**{zscore_val:+.2f}σ**) "
                            f"posiziona l'azienda **sopra la media europea** per singola PMI "
                            f"nel settore **{nace_label(val_settore)}**, ma il gap rispetto "
                            f"al benchmark ML previsto per **{geo_label(val_paese)}** è "
                            f"**{gap_pct:+.1f}%**. "
                            "Questo accade quando il modello prevede un benchmark paese-specifico "
                            "insolitamente elevato (possibile estrapolazione su dati limitati). "
                            f"**Interpretazione consigliata**: il valore dell'azienda è competitivo "
                            f"nella media europea{_p95_str}. "
                            "Verifica il benchmark ML con il range P5–P95 mostrato sopra come "
                            "riferimento alternativo più stabile."
                        )
                    else:
                        st.info(
                            f"ℹ️ L'azienda è **in linea con la media** del settore "
                            f"(gap: **{gap_pct:+.1f}%**{_zstr}). "
                            "Le performance sono allineate al benchmark. Esistono margini di "
                            "miglioramento identificabili attraverso l'ottimizzazione delle "
                            "leve operative."
                        )
                else:
                    st.error(
                        f"⚠️ L'azienda si trova **al di sotto del benchmark per-PMI** "
                        f"del **{abs(gap_pct):.1f}%**{_zstr}. "
                        "Questo segnale di **early warning** indica una sottoperformance "
                        f"rispetto alla media storica delle PMI nel settore **{nace_label(val_settore)}**. "
                        "Si raccomanda un'analisi dei driver di ricavo e un confronto "
                        "con le aziende nel cluster di Alta Performance."
                    )

            # ── Contesto di settore/paese ──────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📊 Contesto di mercato")
            _ctx_df = df_full[
                (df_full["indic_sb"] == val_indicatore) &
                (df_full["nace_r2"]  == val_settore)
            ]["OBS_VALUE"].dropna()
            _ctx_paese = df_full[
                (df_full["indic_sb"] == val_indicatore) &
                (df_full["nace_r2"]  == val_settore) &
                (df_full["geo"]      == val_paese)
            ]["OBS_VALUE"].dropna()
            if len(_ctx_df) > 0 and len(_ctx_paese) > 0:
                _ctx_media_eu    = float(_ctx_df.mean())
                _ctx_media_paese = float(_ctx_paese.mean())
                _ctx_n_paesi     = df_full[
                    (df_full["indic_sb"] == val_indicatore) &
                    (df_full["nace_r2"]  == val_settore)
                ]["geo"].nunique()
                _ctx_label_paese = GEO_LABELS.get(val_paese, val_paese)
                _ctx_label_sett  = NACE_LABELS.get(val_settore, val_settore)

                # ── Riga 1: aggregati di settore (scala paese) ─────────────────
                st.caption(
                    "📦 **Aggregato di settore** (totale per paese, non per singola PMI):"
                )
                cxa, cxb = st.columns(2)
                cxa.metric(
                    f"Media europea – {IND_LABELS[val_indicatore]} (aggregato)",
                    f"{_ctx_media_eu:,.0f} M€",
                    help=(f"Media storica dell'aggregato di settore di {IND_LABELS[val_indicatore]} "
                          f"nel settore {_ctx_label_sett} su tutti i paesi Eurostat disponibili "
                          f"(2011–2020). ⚠️ Scala aggregata: non comparabile con il benchmark "
                          "per-PMI o con il valore inserito.")
                )
                cxb.metric(
                    f"Media storica – {_ctx_label_paese} (aggregato)",
                    f"{_ctx_media_paese:,.0f} M€",
                    help=(f"Media storica dell'aggregato di {IND_LABELS[val_indicatore]} "
                          f"nel settore {_ctx_label_sett} per {_ctx_label_paese} (2011–2020). "
                          "⚠️ Scala aggregata: non comparabile con il benchmark per-PMI.")
                )

                # ── Riga 2: valori per singola PMI (comparabili con il benchmark) ─
                if _rng_settore and "pe_mediana" in _rng_settore:
                    # Calcola la media per-PMI del paese selezionato
                    _ctx_pe_paese_df = df_full[
                        (df_full["indic_sb"] == val_indicatore) &
                        (df_full["nace_r2"]  == val_settore) &
                        (df_full["geo"]      == val_paese)
                    ].merge(
                        df_full[(df_full["indic_sb"] == "V11110") &
                                (df_full["nace_r2"]  == val_settore) &
                                (df_full["geo"]      == val_paese)]
                        [["TIME_PERIOD", "OBS_VALUE"]].rename(
                            columns={"OBS_VALUE": "n_ent"}),
                        on="TIME_PERIOD", how="left"
                    )
                    _ctx_pe_paese_df["n_ent"] = pd.to_numeric(
                        _ctx_pe_paese_df["n_ent"], errors="coerce").replace(0, np.nan)
                    _ctx_pe_paese_df["val_pe"] = (
                        _ctx_pe_paese_df["OBS_VALUE"] / _ctx_pe_paese_df["n_ent"])
                    _ctx_pe_paese_mean = _ctx_pe_paese_df["val_pe"].dropna()

                    st.caption(
                        "🏭 **Per singola PMI** (scala direttamente confrontabile "
                        "con il benchmark ML e il valore inserito):"
                    )
                    cxc, cxd = st.columns(2)
                    cxc.metric(
                        f"Mediana europea per PMI – {IND_LABELS[val_indicatore]}",
                        f"{_rng_settore['pe_mediana']:,.2f} M€",
                        help=(f"Mediana storica del valore per singola PMI di "
                              f"{IND_LABELS[val_indicatore]} nel settore {_ctx_label_sett} "
                              f"(tutti i paesi, 2011–2020). "
                              "Stessa scala del benchmark ML e del valore inserito.")
                    )
                    if len(_ctx_pe_paese_mean) > 0:
                        _val_pe_paese = float(_ctx_pe_paese_mean.mean())
                        _pe_icon = "🟢" if _val_pe_paese >= _rng_settore["pe_mediana"] else "🔴"
                        cxd.metric(
                            f"Media storica per PMI – {_ctx_label_paese}",
                            f"{_val_pe_paese:,.2f} M€",
                            help=(f"Media storica del valore per singola PMI di "
                                  f"{IND_LABELS[val_indicatore]} per {_ctx_label_paese} "
                                  f"nel settore {_ctx_label_sett} (2011–2020).")
                        )
                    else:
                        _pe_icon = "⚪"
                        cxd.metric(f"Media storica per PMI – {_ctx_label_paese}",
                                   "N/D",
                                   help="Dati V11110 non disponibili per questo paese-settore.")

                _pos_icon = "🟢" if _ctx_media_paese > _ctx_media_eu else "🔴"
                st.info(
                    f"{_pos_icon} **{_ctx_label_paese}** si posiziona storicamente "
                    f"{'**sopra**' if _ctx_media_paese >= _ctx_media_eu else '**sotto**'} "
                    f"la media europea per **{IND_LABELS[val_indicatore]}** (aggregato) "
                    f"nel settore **{_ctx_label_sett}** "
                    f"({_ctx_n_paesi} paesi nel dataset). "
                    f"Il benchmark ML si basa su questi dati storici."
                )

            st.caption(
                f"Analisi: {pay_v['name']} | Cutoff training: "
                f"{pay_v.get('anno_cutoff', 'N/D')} | "
                f"Fonte: Eurostat SBS (PMI) | "
                f"Benchmark per-PMI = aggregato ÷ {n_ent_v:,.0f} imprese"
                if n_ent_v else
                f"Analisi: {pay_v['name']} | Cutoff: {pay_v.get('anno_cutoff', 'N/D')} | "
                "Fonte: Eurostat SBS (PMI)"
            )
