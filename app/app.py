"""
app.py — Application Streamlit : Prédiction du Churn TélécomGuinée
Lancer avec : streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

# Ajouter src/ au path pour importer predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor — TélécomGuinée",
    page_icon="📱",
    layout="wide",  # mise en page optimisee,
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A5F, #2563EB);
        color: white; padding: 20px 24px; border-radius: 12px;
        margin-bottom: 24px;
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin: 0; }
    .main-header p  { color: #BDD7FF; margin: 4px 0 0; font-size: 0.95rem; }

    .risk-low    { background:#d1fae5; border-left:5px solid #16a34a;
                   padding:16px; border-radius:8px; }
    .risk-medium { background:#fef3c7; border-left:5px solid #d97706;
                   padding:16px; border-radius:8px; }
    .risk-high   { background:#fee2e2; border-left:5px solid #dc2626;
                   padding:16px; border-radius:8px; }
    .metric-card { background:#f8fafc; border:1px solid #e2e8f0;
                   border-radius:8px; padding:12px 16px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Chargement du modèle ──────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_assets():
    model_path  = os.path.join(MODELS_DIR, 'best_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    if not os.path.exists(model_path):
        return None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📱 TélécomGuinée — Prédiction du Churn Client</h1>
  <p>Identifiez les abonnés à risque de résiliation et déclenchez des offres personnalisées</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar : informations projet ────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Master%201-Fouille%20de%20Donn%C3%A9es-blue",
             use_container_width=True)
    st.markdown("### 📚 Projet P01")
    st.markdown("""
    **Université Kofi Annan de Guinée**
    Master 1 — Systèmes d'Information

    **Étudiant :** Mamadou Bachir Diallo
    **Enseignant :** Y. V. Traoré
    **Méthode :** CRISP-DM

    ---
    **Modèle :** XGBoost (meilleur F1)
    **Dataset :** 15 000 abonnés
    **Taux de churn :** ~24%
    ---
    """)

    if model is None:
        st.error("⚠️ Modèle non trouvé. Exécutez d'abord `python src/train.py`")

# ── Onglets principaux ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Prédiction individuelle",
    "📊 Analyse du dataset",
    "📈 Performance du modèle"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prédiction individuelle
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Saisir les informations du client")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Profil client**")
        sexe           = st.selectbox("Sexe", ["Homme", "Femme"])
        age            = st.slider("Âge", 18, 70, 32)
        region         = st.selectbox("Région", [
            "Conakry", "Kankan", "Boké", "Faranah",
            "N'Zérékoré", "Kindia", "Mamou", "Labé"])
        revenu         = st.number_input("Revenu estimé (GNF)", 100000, 5000000,
                                          450000, step=50000)
        anciennete     = st.slider("Ancienneté (mois)", 1, 96, 12)

    with col2:
        st.markdown("**📋 Abonnement**")
        type_abo       = st.selectbox("Type d'abonnement", ["Prépayé", "Postpayé"])
        moyen_pmt      = st.selectbox("Moyen de paiement",
                                       ["Orange Money", "Espèces",
                                        "Virement bancaire", "Carte bancaire"])
        recharge       = st.number_input("Recharge mensuelle moy. (GNF)",
                                          5000, 500000, 45000, step=5000)
        forfait_int    = st.radio("Forfait international", ["Non", "Oui"],
                                   horizontal=True)
        msg_vocale     = st.radio("Messagerie vocale", ["Oui", "Non"],
                                   horizontal=True)
        retard         = st.slider("Retard paiement (jours)", 0, 30, 0)

    with col3:
        st.markdown("**📡 Usage**")
        min_jour       = st.slider("Minutes/jour (appels)", 0, 450, 150)
        min_nuit       = st.slider("Minutes/nuit (appels)", 0, 300, 80)
        min_int        = st.slider("Minutes internationales", 0, 120, 2)
        donnees        = st.slider("Données (Mo)", 50, 6500, 1500)
        nb_sms         = st.slider("Nombre de SMS", 0, 80, 8)
        appels_svc     = st.slider("Appels service client", 0, 15, 2)
        pannes         = st.slider("Pannes signalées (30j)", 0, 8, 0)
        reclamations   = st.number_input("Nombre total réclamations", 0, 20,
                                          int(appels_svc + pannes))

    # ── Prédiction ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔮 Prédire le risque de churn", type="primary",
                 use_container_width=True):

        if model is None:
            st.error("Modèle non chargé — exécutez d'abord `python src/train.py`")
        else:
            client = {
                'age': age, 'anciennete_mois': anciennete,
                'revenu_estime_gnf': revenu,
                'recharge_mensuelle_moy_gnf': recharge,
                'minutes_jour': min_jour, 'minutes_nuit': min_nuit,
                'minutes_internationales': min_int,
                'donnees_mo': donnees, 'nombre_sms': nb_sms,
                'appels_service_client': appels_svc,
                'pannes_signalees_30j': pannes,
                'nombre_reclamations': int(reclamations),
                'retard_paiement_jours': retard,
                'sexe': sexe, 'type_abonnement': type_abo,
                'forfait_international': forfait_int,
                'messagerie_vocale': msg_vocale,
                'region': region, 'moyen_paiement': moyen_pmt,
            }

            try:
                from predict import predict_churn
                result = predict_churn(client)

                proba   = result['probabilite']
                segment = result['segment_risque']
                action  = result['recommandation']

                # Jauge de risque
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(proba * 100, 1),
                    title={'text': "Probabilité de churn (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#dc2626" if proba > 0.6
                                         else "#d97706" if proba > 0.4
                                         else "#16a34a"},
                        'steps': [
                            {'range': [0, 40],  'color': '#d1fae5'},
                            {'range': [40, 60], 'color': '#fef3c7'},
                            {'range': [60, 80], 'color': '#fee2e2'},
                            {'range': [80, 100],'color': '#fecaca'},
                        ],
                        'threshold': {'line': {'color': 'black', 'width': 3},
                                      'thickness': 0.75, 'value': 50}
                    }
                ))
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)

                # Résultat
                css_class = ("risk-high" if proba > 0.6
                             else "risk-medium" if proba > 0.4
                             else "risk-low")
                emoji     = "🔴" if proba > 0.6 else "🟡" if proba > 0.4 else "🟢"

                st.markdown(f"""
                <div class="{css_class}">
                  <h3>{emoji} {segment} — {round(proba*100,1)}% de risque de churn</h3>
                  <p><strong>Action recommandée :</strong> {action}</p>
                  <p><em>Ratio data/voix calculé :
                  {round(donnees / max(1, min_jour+min_nuit+min_int), 2)}</em></p>
                </div>
                """, unsafe_allow_html=True)

                # Coût métier
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"💸 Coût rétention estimé : **50 000 GNF**")
                with col_b:
                    st.error(f"💸 Coût acquisition si perdu : **250 000 GNF**")

            except Exception as e:
                st.error(f"Erreur : {e}")
                st.info("Vérifiez que le modèle a bien été entraîné.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analyse du dataset
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',
                              'guinee_telecom_churn_FR.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['churn_num'] = (df['resiliation'] == 'Oui').astype(int)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total clients",   f"{len(df):,}")
        c2.metric("Taux de churn",   f"{df['churn_num'].mean()*100:.1f}%")
        c3.metric("Ancienneté moy.", f"{df['anciennete_mois'].mean():.0f} mois")
        c4.metric("Réclamations moy.",f"{df['nombre_reclamations'].mean():.1f}")

        col_l, col_r = st.columns(2)

        with col_l:
            # Churn par région
            churn_region = df.groupby('region')['churn_num'].mean().reset_index()
            churn_region.columns = ['region', 'taux_churn']
            churn_region = churn_region.sort_values('taux_churn', ascending=True)
            fig1 = px.bar(churn_region, x='taux_churn', y='region',
                          orientation='h', title="Taux de churn par région",
                          color='taux_churn', color_continuous_scale='RdYlGn_r')
            fig1.update_layout(height=350)
            st.plotly_chart(fig1, use_container_width=True)

        with col_r:
            # Distribution ancienneté selon churn
            fig2 = px.histogram(df, x='anciennete_mois', color='resiliation',
                                 barmode='overlay', nbins=40,
                                 title="Ancienneté selon résiliation",
                                 color_discrete_map={'Non':'#16a34a','Oui':'#dc2626'})
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            # Churn par type abonnement
            fig3 = px.pie(df, names='type_abonnement', color='type_abonnement',
                           title="Répartition par type d'abonnement",
                           color_discrete_map={'Prépayé':'#2563EB','Postpayé':'#7C3AED'})
            st.plotly_chart(fig3, use_container_width=True)

        with col_r2:
            # Réclamations vs churn
            fig4 = px.box(df, x='resiliation', y='nombre_reclamations',
                           color='resiliation',
                           title="Réclamations selon résiliation",
                           color_discrete_map={'Non':'#16a34a','Oui':'#dc2626'})
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Dataset non trouvé dans `data/raw/`.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Performance du modèle
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    results_path = os.path.join(MODELS_DIR, 'resultats_modeles.json')
    if os.path.exists(results_path):
        import json
        with open(results_path) as f:
            results = json.load(f)

        st.subheader("Tableau comparatif des modèles")
        rows = []
        for name, r in results.items():
            rows.append({
                'Modèle': name.replace('_', ' '),
                'Accuracy':  r['accuracy'],
                'Précision': r['precision'],
                'Rappel':    r['recall'],
                'F1-Score':  r['f1'],
                'ROC-AUC':   r['roc_auc'],
            })
        df_res = pd.DataFrame(rows).set_index('Modèle')
        best_f1 = df_res['F1-Score'].idxmax()
        st.dataframe(df_res.style.highlight_max(axis=0, color='#d1fae5'), height=200)
        st.success(f"✅ Meilleur modèle : **{best_f1}** (F1 = {df_res.loc[best_f1,'F1-Score']})")

        # Graphique radar
        categories = ['Accuracy', 'Précision', 'Rappel', 'F1-Score', 'ROC-AUC']
        fig_radar = go.Figure()
        colors = ['#2563EB', '#16A34A', '#D97706', '#BE185D']
        for i, (name, r) in enumerate(results.items()):
            vals = [r['accuracy'], r['precision'], r['recall'],
                    r['f1'], r['roc_auc']]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill='toself', name=name.replace('_', ' '),
                line_color=colors[i % len(colors)]))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Comparaison des modèles (radar)", height=420)
        st.plotly_chart(fig_radar, use_container_width=True)

        # Contexte métier du rappel
        st.markdown("### 💡 Interprétation métier")
        st.info("""
        **Pourquoi prioriser le Rappel ?**

        Un **Faux Négatif** (churner non détecté) coûte 250 000 GNF (coût d'acquisition).
        Un **Faux Positif** (client fidèle contacté par erreur) coûte 50 000 GNF (offre inutile).

        Le ratio étant de 1:5, on préfère maximiser le **Rappel** (capturer le maximum de vrais churners)
        même au prix de quelques faux positifs.
        """)
    else:
        st.warning("Modèle non encore entraîné. Exécutez `python src/train.py`.")
        st.info("Une fois l'entraînement terminé, revenez ici pour voir les résultats.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#64748b;font-size:0.85rem'>"
    "P01 — Prédiction Churn TélécomGuinée · "
    "Master 1 SI — Université Kofi Annan de Guinée · "
    "Mamadou Bachir Diallo</p>",
    unsafe_allow_html=True)
