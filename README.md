# P01 — Prédiction du Churn des Abonnés d'un Opérateur Mobile Guinéen

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

## Contexte et problématique métier

TélécomGuinée SA perd environ **18% de ses abonnés par trimestre**, soit 720 000 clients sur 4 millions. Le coût d'acquisition d'un nouveau client est **5 fois supérieur** au coût de rétention d'un client existant.

**Objectif** : construire un modèle de classification binaire capable d'identifier les clients susceptibles de résilier dans les **30 prochains jours**, afin de déclencher automatiquement des offres de rétention personnalisées.

**Utilisateurs du modèle** :
- Équipe commerciale → liste quotidienne des clients à risque élevé
- Système CRM → déclenchement automatique d'offres (SMS, appels, remises)
- Direction → tableau de bord KPI rétention

## Source et description des données

| Attribut | Valeur |
|---|---|
| Fichier | `data/raw/guinee_telecom_churn_FR.csv` |
| Lignes | 15 000 |
| Colonnes | 23 |
| Variable cible | `resiliation` (Oui / Non) — déséquilibre ~24% Oui |
| Source | Données synthétiques générées avec numpy/random |

### Dictionnaire des variables

| Variable | Type | Description |
|---|---|---|
| id_client | str | Identifiant unique client |
| region | str | Région de Guinée (Conakry, Kankan, Boké…) |
| sexe | str | Homme / Femme |
| age | int | Âge du client (18–70 ans) |
| revenu_estime_gnf | int | Revenu estimé en Francs Guinéens |
| anciennete_mois | int | Durée d'abonnement en mois (1–96) |
| type_abonnement | str | Prépayé / Postpayé |
| forfait_international | str | Oui / Non |
| messagerie_vocale | str | Oui / Non |
| recharge_mensuelle_moy_gnf | int | Recharge mensuelle moyenne en GNF |
| moyen_paiement | str | Orange Money / Espèces / Virement / Carte |
| minutes_jour | int | Minutes d'appels en journée |
| minutes_nuit | int | Minutes d'appels la nuit |
| minutes_internationales | int | Minutes d'appels internationaux |
| donnees_mo | int | Consommation data en Mo |
| ratio_data_voix | float | donnees_mo / minutes_totales |
| nombre_sms | int | Nombre de SMS envoyés |
| appels_service_client | int | Appels au service client |
| pannes_signalees_30j | int | Pannes signalées sur 30 jours |
| nombre_reclamations | int | Total réclamations (pannes + appels SAV) |
| retard_paiement_jours | int | Jours de retard de paiement |
| retour_client | str | Commentaire libre |
| resiliation | str | **Variable cible** : Oui / Non |


## Méthodologie (CRISP-DM)

```
1. Compréhension métier    → Définir le coût d'un faux négatif vs faux positif
2. Compréhension données   → EDA, distributions, corrélations, valeurs manquantes
3. Préparation             → Encodage, normalisation, SMOTE
4. Modélisation            → Régression Logistique, Random Forest, XGBoost
5. Évaluation              → F1, ROC-AUC, matrice de confusion, lift curve
6. Déploiement             → Application Streamlit
```

## Résultats — Tableau comparatif des modèles

| Modèle | Accuracy | Précision | Rappel | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Régression Logistique** | 0.593 | 0.312 | **0.577** | **0.405** | 0.637 |
| Random Forest | 0.667 | 0.332 | 0.385 | 0.357 | 0.630 |
| XGBoost | 0.675 | 0.339 | 0.374 | 0.356 | 0.630 |
| XGBoost + SMOTE | 0.760 | 0.500 | 0.124 | 0.198 | 0.636 |

> **Meilleur modèle : Régression Logistique** (F1=0.405, Rappel=0.577)
> Le Rappel est priorisé car un faux négatif (churner manqué) coûte 250 000 GNF vs 50 000 GNF pour un faux positif.

## Variables les plus explicatives

1. `anciennete_mois`  Les clients récents (< 6 mois) churne 3× plus
2. `pannes_signalees_30j` Chaque panne augmente le risque de +12%
3. `retard_paiement_jours` Signal fort d'insatisfaction globale
4. `appels_service_client` Proxy de l'insatisfaction
5. `nombre_reclamations` — très prédictive (confirmée)

## Limites et pistes d'amélioration

- Dataset synthétique : les patterns peuvent être trop réguliers vs données réelles
- Absence de données géographiques fines (couverture réseau par quartier)
- Piste : intégrer les données CDR réelles et un modèle LSTM pour capturer la dynamique temporelle

## Comment exécuter le projet

```bash
# 1. Cloner le dépôt
git clone https://github.com/Votre_username/P01-Prediction-du-churn-des-abonnes-d-un-operateur-mobile-guineen.git
cd p01-churn-telecom-guinee

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer les notebooks (dans l'ordre)
jupyter notebook notebooks/01_analyse_exploratoire.ipynb
jupyter notebook notebooks/02_modelisation.ipynb

# 5. Lancer l'application Streamlit
streamlit run app/app.py
```

## Lien vers la vidéo YouTube

[Fouille de Données M1 - Prédiction Churn Télécom Guinée - Mamadou Bachir Diallo](https://youtube.com/VOTRE_LIEN)

## Auteur

**Mamadou Bachir Diallo**
Master 1 Systèmes d'Information  Université Kofi Annan de Guinée
Année académique 2025-2026

