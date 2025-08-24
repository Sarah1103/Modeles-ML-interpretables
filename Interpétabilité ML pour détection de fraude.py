#Import des librairies
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr 
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Import des librairies pour les modèles avancés
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier; HAS_LGBM = True
except Exception:
    HAS_LGBM = False

#Import des librairies pour le traitement du déséquilibre

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, NearMiss
    from imblearn.pipeline import Pipeline
    HAS_IMB = True
except Exception:
    HAS_IMB = False

df = pd.read_csv("creditcard.csv") #Chargement du dataset
X, y = df.drop("Class", axis=1), df["Class"] #Séparation features/target

# Split (division) du dataset en données train/test 80/20 avec stratify
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Construction du modèle selon le choix de l'utilisateur
def build_model(name, params):
    if name == "RandomForest":
        return RandomForestClassifier(**params)
    if name == "XGBoost" and HAS_XGB:
        return XGBClassifier(**params)
    if name == "LightGBM" and HAS_LGBM:
        return LGBMClassifier(**params)
    if name == "LogisticRegression":
        p = params.copy()
        p.setdefault("solver", "lbfgs")
        p.setdefault("max_iter", 1000)
        p.setdefault("n_jobs", -1)
        try:
            return LogisticRegression(**p)
        except TypeError:
            p.pop("n_jobs", None)
            return LogisticRegression(**p)
    raise ValueError("Modèle non disponible")

#Construction de l'échantillineur selon le choix de l'utilisateur
def build_sampler(kind, method):
    if not HAS_IMB or kind == "Aucun":
        return None
    if kind == "Oversampling":
        return {"SMOTE": SMOTE, "ADASYN": ADASYN, "BorderlineSMOTE": BorderlineSMOTE}[method](random_state=42)
    else:
        return RandomUnderSampler(random_state=42) if method == "RandomUnderSampler" else NearMiss(version=1)

# SHAP explainer (interprétabilité)
def get_explainer(model, model_name):
    bg = Xtr.sample(min(len(Xtr), 1000), random_state=0)
    if model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        return shap.TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="probability")
    else:
        try:
            return shap.LinearExplainer(model, bg, feature_perturbation="interventional")
        except Exception:
            return shap.KernelExplainer(lambda d: model.predict_proba(pd.DataFrame(d, columns=X.columns))[:, 1], bg)
# Pour SHAP, on prend toujours la classe 1 (fraude)
def shap_cls1(sv):
    if isinstance(sv, list):
        return sv[1] if len(sv) > 1 else sv[0]
    if isinstance(sv, np.ndarray) and sv.ndim == 3 and sv.shape[-1] == 2:
        return sv[:, :, 1]
    return sv
# Importance des features
def feature_importances_df(fitted):
    if hasattr(fitted, "feature_importances_"):
        return pd.DataFrame({"Feature": X.columns, "Importance": fitted.feature_importances_}).sort_values("Importance", ascending=False)
    if hasattr(fitted, "coef_") and getattr(fitted, "coef_", None) is not None:
        coefs = np.abs(fitted.coef_[0]) if fitted.coef_.ndim == 2 else np.abs(fitted.coef_)
        return pd.DataFrame({"Feature": X.columns, "Importance": coefs}).sort_values("Importance", ascending=False)
    return pd.DataFrame({"Feature": X.columns, "Importance": np.full(len(X.columns), 1.0/len(X.columns))})
# KPI HTML
def kpi_html(label, value, color="#000"):
    return f"<div style='text-align:center'><div style='font-size:16px'>{label}</div><div style='font-size:42px;color:{color};font-weight:700'>{value}</div></div>"
# Correlation heatmap
def correlation_heatmap(cols):
    corr = X[cols].corr(method="pearson")
    fig = go.Figure(go.Heatmap(z=corr.values, x=cols, y=cols, colorscale="RdBu", zmin=-1, zmax=1, colorbar=dict(title="corr")))
    fig.update_layout(title="Matrice de corrélation (top 15)", height=520, margin=dict(l=20, r=20, t=40, b=20))
    return fig
# Distribution des valeurs des variables par classe
def two_hist_plot(df_source, feature):
    # Prepare data
    s0 = pd.to_numeric(df_source.loc[df_source["Class"] == 0, feature], errors="coerce").dropna()
    s1 = pd.to_numeric(df_source.loc[df_source["Class"] == 1, feature], errors="coerce").dropna()

    xmin = float(min(s0.min() if len(s0) else 0, s1.min() if len(s1) else 0))
    xmax = float(max(s0.max() if len(s0) else 1, s1.max() if len(s1) else 1))
    if xmin == xmax:
        xmax = xmin + 1.0
    bins = 60 if feature != "Amount" else 50
    step = (xmax - xmin) / max(bins, 1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Non frauduleuses", "Frauduleuses"))
    fig.add_trace(go.Histogram(x=s0, xbins=dict(start=xmin, end=xmax, size=step), marker_color="#1f77b4", showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=s1, xbins=dict(start=xmin, end=xmax, size=step), marker_color="#d62728", showlegend=False), row=1, col=2)
    fig.update_layout(title=f"Distribution de {feature}", height=360, bargap=0.05, margin=dict(l=20, r=20, t=40, b=10))
    fig.update_xaxes(title_text=feature, row=1, col=1)
    fig.update_xaxes(title_text=feature, row=1, col=2)
    return fig

#Interface Gradio
with gr.Blocks(title="Fraude Bancaire – Dashboard") as demo:
    gr.Markdown("# Paramétrer le modèle de détection de fraude")

    # Chargement d'un autre dataset
    gr.Markdown("## Dataset")
    with gr.Row():
        file_input = gr.File(label="Charger un fichier CSV ou XLSX (doit contenir une colonne 'Class')", file_types=[".csv", ".xlsx"], type="filepath")
        btn_load = gr.Button("Charger ce dataset")
    ds_status = gr.Markdown("**Dataset actuel :** `creditcard.csv` (par défaut).")

    # Model + sampling
    with gr.Row():
        model_choices = ["RandomForest", "LogisticRegression"] + (["XGBoost"] if HAS_XGB else []) + (["LightGBM"] if HAS_LGBM else [])
        dd_model = gr.Dropdown(model_choices, value="RandomForest", label="Modèle")

        dd_sampling_kind = gr.Dropdown(["Aucun", "Oversampling", "Undersampling"], value="Aucun", label="Échantillonnage (déséquilibre)")
        dd_sampling_method = gr.Dropdown(choices=["-"], value="-", label="Méthode", interactive=False)

    # Paramètres par modèle
    with gr.Group(visible=True) as grp_rf:
        with gr.Accordion("RandomForest – paramètres", open=True):
            rf_n_estimators = gr.Number(value=100, label="n_estimators", precision=0)
            rf_max_depth = gr.Number(value=15, label="max_depth (0 = None)", precision=0)
            rf_min_split = gr.Number(value=5, label="min_samples_split", precision=0)
            rf_min_leaf = gr.Number(value=2, label="min_samples_leaf", precision=0)
            rf_bootstrap = gr.Checkbox(value=True, label="bootstrap")
            rf_class_weight = gr.Dropdown([None, "balanced", "balanced_subsample"], value="balanced", label="class_weight")
    # XGBoost
    with gr.Group(visible=False) as grp_xgb:
        with gr.Accordion("XGBoost – paramètres", open=True):
            xgb_n_estimators = gr.Number(value=300, label="n_estimators", precision=0)
            xgb_max_depth = gr.Number(value=6, label="max_depth", precision=0)
            xgb_lr = gr.Number(value=0.1, label="learning_rate")
            xgb_subsample = gr.Number(value=0.8, label="subsample")
            xgb_colsample = gr.Number(value=0.8, label="colsample_bytree")
            xgb_spw = gr.Number(value=10.0, label="scale_pos_weight")
    # LightGBM
    with gr.Group(visible=False) as grp_lgbm:
        with gr.Accordion("LightGBM – paramètres", open=True):
            lgbm_n_estimators = gr.Number(value=300, label="n_estimators", precision=0)
            lgbm_num_leaves = gr.Number(value=31, label="num_leaves", precision=0)
            lgbm_lr = gr.Number(value=0.1, label="learning_rate")
            lgbm_subsample = gr.Number(value=0.8, label="subsample")
            lgbm_colsample = gr.Number(value=0.8, label="colsample_bytree")
            lgbm_cw = gr.Dropdown([None, "balanced"], value="balanced", label="class_weight")
    # Logistic Regression
    with gr.Group(visible=False) as grp_lr:
        with gr.Accordion("LogisticRegression – paramètres", open=True):
            lr_C = gr.Number(value=1.0, label="C")
            lr_max_iter = gr.Number(value=1000, label="max_iter", precision=0)
            lr_penalty = gr.Dropdown(["l2", "none"], value="l2", label="penalty")
            lr_cw = gr.Dropdown([None, "balanced"], value="balanced", label="class_weight")
    # Bouton d'entraînement
    btn_run = gr.Button(" Entraîner et afficher", variant="primary")

    # Résultats de la fraude
    gr.Markdown("## Résultats de détection de fraude du modèle entraîné")
    with gr.Row():
        kpi_f = gr.HTML()
        kpi_n = gr.HTML()
        kpi_t = gr.HTML()
        kpi_m = gr.HTML()

    with gr.Row():
        metrics_df = gr.Dataframe(label="Scores de l’algorithme (sur test)", interactive=False)
        metrics_defs = gr.Markdown("""
**Définitions (fraude = classe 1)**  
- **Accuracy** : part de prédictions correctes (toutes classes).  
- **Precision** : parmi les *prédictions* de fraude, part réellement fraude (↓ faux positifs).  
- **Recall** : parmi les fraudes réelles, part détectée (↓ faux négatifs).  
- **F1-score** : moyenne harmonique Precision/Recall (équilibre).  
- **AUC** : aire sous la courbe ROC, probabilité qu’une fraude soit scorée au-dessus d’une non-fraude.
        """)

    # Titres + graphes
    gr.Markdown("### Importance des variables")
    fig_imp = gr.Plot(label="Importance des variables (top 15)")

    gr.Markdown("### Matrice de corrélation (top 15)")
    fig_corr = gr.Plot(label="Matrice de corrélation (top 15)")

    # Distributions top 5
    gr.Markdown("### Distribution des 5 variables les plus importantes")
    gr.Markdown("*(Pour chaque variable, on observe ses valeurs quand la transaction est frauduleuse et quand elle est normale.)*")
    dd_feat = gr.Dropdown(choices=[], value=None, label="Choisissez une variable (top 5)")
    fig_dist = gr.Plot()

    # SHAP summary + waterfall
    gr.Markdown("### SHAP — Summary plot (global)")
    gr.Markdown("Le summary plot montre, pour toutes les transactions, quelles variables influencent le plus la **probabilité prédite de fraude** (points rouges = + fraude, bleus = – fraude).")
    shap_summary_plot = gr.Plot()
    shap_summary_read = gr.Markdown(
        "Lecture : l’axe vertical liste les variables (les plus importantes en haut). "
        "L’axe horizontal est l’impact SHAP sur la probabilité prédite. "
        "Valeurs SHAP < 0 → vers « non fraude », > 0 → vers « fraude ». "
        "La couleur reflète la valeur réelle de la variable (bleu = faible, rouge = élevée)."
    )

    gr.Markdown("### SHAP — Waterfall (local)")
    gr.Markdown("Sélectionnez une transaction **détectée** comme fraude pour voir comment chaque variable a poussé la probabilité **vers le haut** ou **vers le bas**.")
    with gr.Row():
        dd_idx = gr.Dropdown(choices=[], value=None, label="Index de transaction prédite fraude")
        btn_waterfall = gr.Button("Afficher le Waterfall")
    shap_waterfall_plot = gr.Plot()
    shap_waterfall_read = gr.Markdown(
        "Lecture : Le Waterfall plot SHAP montre, pour cette transaction, comment chaque variable "
        "contribue à la prédiction finale. On part d’une valeur de base (la valeur de E[f(X)]) en bas, "
        "puis chaque variable ajoute ou enlève un effet : si la valeur SHAP est négative (barre bleue), "
        "elle pousse la prédiction vers « non fraude », si elle est positive (barre rouge), "
        "elle la pousse vers « fraude ». Le cumul de ces contributions permet d’atteindre la "
        "prédiction finale affichée à droite."
    )

    # Table des fraudes détectées
    gr.Markdown("### Transactions frauduleuses détectées (échantillon)")
    table_frauds = gr.Dataframe(interactive=False)

    
    st_pipe = gr.State()
    st_expl = gr.State()
    st_df_pred = gr.State()
    st_top5 = gr.State()

    # Nouveau : callback de chargement dataset
    def load_dataset(file_path):
        """
        Charge un CSV/XLSX, vérifie la colonne 'Class', met à jour les globals df/X/y/Xtr...
        Si erreur, on garde l'ancien dataset et on affiche un message.
        """
        global df, X, y, Xtr, Xte, ytr, yte
        if file_path is None:
            return "**Dataset actuel :** `creditcard.csv` (par défaut)."

        path = str(file_path)
        try:
            if path.lower().endswith(".csv"):
                new_df = pd.read_csv(path)
            else:
                new_df = pd.read_excel(path)
        except Exception as e:
            return f"❌ Erreur de lecture du fichier : {e}"

        if "Class" not in new_df.columns:
            return "❌ Le fichier chargé ne contient pas la colonne **'Class'**. Dataset conservé (creditcard.csv)."

        # MàJ des globals
        df = new_df.copy()
        # Nettoyage léger : supprime les colonnes entièrement vides
        df.dropna(axis=1, how="all", inplace=True)
        # Remplace les inf par NaN puis dropna 
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=0, how="any", inplace=True)

        X_cols = [c for c in df.columns if c != "Class"]
        if len(X_cols) == 0:
            return "❌ Aucune feature disponible (seule 'Class' est présente). Dataset conservé."

        try:
            X_new, y_new = df[X_cols], df["Class"].astype(int)
        except Exception:
            return "❌ Impossible de caster 'Class' en entier (0/1). Vérifie les valeurs. Dataset conservé."

        try:
            Xtr_new, Xte_new, ytr_new, yte_new = train_test_split(
                X_new, y_new, test_size=0.2, stratify=y_new, random_state=42
            )
        except Exception:
            # En cas de stratify impossible, split simple
            Xtr_new, Xte_new, ytr_new, yte_new = train_test_split(
                X_new, y_new, test_size=0.2, random_state=42
            )

        # Applique
        globals()["X"], globals()["y"] = X_new, y_new
        globals()["Xtr"], globals()["Xte"] = Xtr_new, Xte_new
        globals()["ytr"], globals()["yte"] = ytr_new, yte_new

        return f"✅ Dataset chargé : `{path.split('/')[-1]}` — {len(df):,} lignes, {len(X_new.columns)} variables (dont 'Class').".replace(",", " ")

    btn_load.click(load_dataset, inputs=file_input, outputs=ds_status)

    # Callbacks existants
    def on_model_change(m):
        return (
            gr.update(visible=(m == "RandomForest")),
            gr.update(visible=(m == "XGBoost")),
            gr.update(visible=(m == "LightGBM")),
            gr.update(visible=(m == "LogisticRegression")),
        )
    dd_model.change(on_model_change, dd_model, [grp_rf, grp_xgb, grp_lgbm, grp_lr])

    def on_sampling_kind(kind):
        if not HAS_IMB:
            return gr.update(choices=["-"], value="-", interactive=False)
        if kind == "Oversampling":
            return gr.update(choices=["SMOTE", "ADASYN", "BorderlineSMOTE"], value="SMOTE", interactive=True)
        if kind == "Undersampling":
            return gr.update(choices=["RandomUnderSampler", "NearMiss"], value="RandomUnderSampler", interactive=True)
        return gr.update(choices=["-"], value="-", interactive=False)
    dd_sampling_kind.change(on_sampling_kind, dd_sampling_kind, dd_sampling_method)

    def train_and_show(
        model_name, sampling_kind, sampling_method,
        rf_n_estimators, rf_max_depth, rf_min_split, rf_min_leaf, rf_bootstrap, rf_class_weight,
        xgb_n_estimators, xgb_max_depth, xgb_lr, xgb_subsample, xgb_colsample, xgb_spw,
        lgbm_n_estimators, lgbm_num_leaves, lgbm_lr, lgbm_subsample, lgbm_colsample, lgbm_cw,
        lr_C, lr_max_iter, lr_penalty, lr_cw
    ):
        # Params par model
        if model_name == "RandomForest":
            params = dict(
                n_estimators=int(rf_n_estimators),
                max_depth=None if int(rf_max_depth) == 0 else int(rf_max_depth),
                min_samples_split=int(rf_min_split),
                min_samples_leaf=int(rf_min_leaf),
                bootstrap=bool(rf_bootstrap),
                class_weight=rf_class_weight,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "XGBoost":
            params = dict(
                n_estimators=int(xgb_n_estimators),
                max_depth=int(xgb_max_depth),
                learning_rate=float(xgb_lr),
                subsample=float(xgb_subsample),
                colsample_bytree=float(xgb_colsample),
                objective="binary:logistic",
                n_jobs=-1, random_state=42,
                scale_pos_weight=float(xgb_spw),
                eval_metric="logloss",
                tree_method="hist"
            )
        elif model_name == "LightGBM":
            params = dict(
                n_estimators=int(lgbm_n_estimators),
                num_leaves=int(lgbm_num_leaves),
                learning_rate=float(lgbm_lr),
                subsample=float(lgbm_subsample),
                colsample_bytree=float(lgbm_colsample),
                objective="binary",
                class_weight=lgbm_cw,
                random_state=42,
                n_jobs=-1
            )
        else:  # LogisticRegression
            params = dict(
                C=float(lr_C),
                max_iter=int(lr_max_iter),
                penalty=lr_penalty,
                class_weight=lr_cw,
                solver="lbfgs",
                n_jobs=-1
            )

        # Construction pipeline
        sampler = build_sampler(sampling_kind, sampling_method)
        model = build_model(model_name, params)
        pipe = Pipeline([("sampler", sampler), ("model", model)]) if (sampler is not None and HAS_IMB) else model

        # Fit (Entraînement)
        pipe.fit(Xtr, ytr)
        fitted = pipe.named_steps["model"] if hasattr(pipe, "named_steps") else pipe

        # KPIs sur les prédictions 
        df_pred = df.copy()
        df_pred["Predicted"] = pipe.predict(X).astype(int)
        n_f = int((df_pred["Predicted"] == 1).sum())
        n_n = int((df_pred["Predicted"] == 0).sum())
        taux = df_pred["Class"].mean() * 100.0
        montant = float(df_pred.loc[df_pred["Predicted"] == 1, "Amount"].sum()) if "Amount" in df_pred.columns else float(0)

        k_f = kpi_html("Fraudes", f"{n_f:,}", "#d62728")
        k_n = kpi_html("Transactions normales", f"{n_n:,}", "#1a7f37")
        k_t = kpi_html("Taux de fraude", f"{taux:.3f}%", "#ff9900")
        k_m = kpi_html("Montant total fraudes", f"{montant:,.0f} €".replace(",", " "), "#6f42c1")

        # Metrics 
        y_pred = pipe.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        prec = precision_score(yte, y_pred, zero_division=0)
        rec = recall_score(yte, y_pred, zero_division=0)
        f1 = f1_score(yte, y_pred, zero_division=0)
        try:
            y_score = pipe.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, y_score)
        except Exception:
            auc = np.nan

        metrics = pd.DataFrame({
            "Métrique": ["Accuracy", "Precision (classe 1)", "Recall (classe 1)", "F1-score (classe 1)", "AUC (ROC)"],
            "Valeur":   [f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{auc:.4f}" if not np.isnan(auc) else "NA"]
        })

        # importance des variables + correlation
        fi_df = feature_importances_df(fitted)
        fig_imp_ = px.bar(fi_df.head(15), x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale="viridis",
                          title="Importance des variables (top 15)")
        fig_imp_.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, margin=dict(l=20, r=20, t=40, b=10))

        top15 = fi_df["Feature"].head(15).tolist()
        fig_corr_ = correlation_heatmap(top15)

        # Distributions (top 5)
        top5 = fi_df["Feature"].head(5).tolist()
        dist_fig_default = two_hist_plot(df, top5[0]) if top5 else go.Figure()

        # SHAP summary
        expl = get_explainer(fitted, model_name)
        Xplot = Xte.sample(2000, random_state=0) if len(Xte) > 2000 else Xte
        try:
            sv = expl.shap_values(Xplot)
        except Exception:
            sv = expl.shap_values(Xplot)  # fallback same
        sv1 = shap_cls1(sv)

        plt.figure()
        plt.title("SHAP — Summary plot (global)")
        shap.summary_plot(sv1, Xplot, show=False)
        fig_shap_sum = plt.gcf()

        # Id des fraudes (predicted=1)
        fraud_idx = df_pred.index[df_pred["Predicted"] == 1].tolist()

        # Frauds table
        fraud_table = df_pred[df_pred["Predicted"] == 1].copy()
        fraud_table.insert(0, "id", fraud_table.index)
        fraud_table = fraud_table[["id"] + [c for c in X.columns]].head(300)

        # Résultats 
        return (
            k_f, k_n, k_t, k_m,                       # KPIs
            metrics,                                   # metrics table
            fig_imp_, fig_corr_,                       # importance + corr
            gr.update(choices=top5, value=(top5[0] if top5 else None)), dist_fig_default,  # distributions
            fig_shap_sum,                               # SHAP summary
            gr.update(choices=fraud_idx, value=(fraud_idx[0] if len(fraud_idx) else None)),  # idx 
            go.Figure(),                                # placeholder waterfall
            fraud_table,                                # frauds table
            pipe, expl, df_pred, top5                   # states
        )

    btn_run.click(
        train_and_show,
        inputs=[
            dd_model, dd_sampling_kind, dd_sampling_method,
            rf_n_estimators, rf_max_depth, rf_min_split, rf_min_leaf, rf_bootstrap, rf_class_weight,
            xgb_n_estimators, xgb_max_depth, xgb_lr, xgb_subsample, xgb_colsample, xgb_spw,
            lgbm_n_estimators, lgbm_num_leaves, lgbm_lr, lgbm_subsample, lgbm_colsample, lgbm_cw,
            lr_C, lr_max_iter, lr_penalty, lr_cw
        ],
        outputs=[
            kpi_f, kpi_n, kpi_t, kpi_m,
            metrics_df,
            fig_imp, fig_corr,
            dd_feat, fig_dist,
            shap_summary_plot,
            dd_idx, shap_waterfall_plot,
            table_frauds,
            st_pipe, st_expl, st_df_pred, st_top5
        ]
    )

    # Mise à jour de la distibution quand l'utilisateur choisit une variable
    def update_distribution(feat, df_pred_state, top5_list):
        if feat is None and top5_list:
            feat = top5_list[0]
        if df_pred_state is None or feat is None:
            return go.Figure()
    
        return two_hist_plot(df, feat)

    dd_feat.change(update_distribution, inputs=[dd_feat, st_df_pred, st_top5], outputs=fig_dist)

    # Affichage du SHAP waterfall pour une transaction choisie
    def show_waterfall(idx, explainer_state, pipe_state):
        if idx is None or explainer_state is None:
            return go.Figure()
        x_one = X.loc[[int(idx)]]
        try:
            sv = explainer_state.shap_values(x_one)
        except Exception:
            sv = explainer_state.shap_values(x_one)
        # Selectionner class 1
        if isinstance(sv, list):
            sv_row = sv[1][0] if len(sv) > 1 else sv[0][0]
            base = explainer_state.expected_value[1] if isinstance(explainer_state.expected_value, (list, np.ndarray)) else explainer_state.expected_value
        elif isinstance(sv, np.ndarray) and sv.ndim == 3 and sv.shape[-1] == 2:
            sv_row = sv[0, :, 1]
            base = explainer_state.expected_value[1]
        else:
            sv_row = sv[0]
            ev = explainer_state.expected_value
            base = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else (ev if np.isscalar(ev) else ev[0])

        plt.figure()
        plt.title("SHAP — Waterfall (local)")
        exp_local = shap.Explanation(values=sv_row, base_values=base, data=x_one.values[0], feature_names=X.columns)
        shap.plots.waterfall(exp_local, max_display=10, show=False)
        return plt.gcf()

    btn_waterfall.click(show_waterfall, inputs=[dd_idx, st_expl, st_pipe], outputs=shap_waterfall_plot)

# ---------- MAIN ----------
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
