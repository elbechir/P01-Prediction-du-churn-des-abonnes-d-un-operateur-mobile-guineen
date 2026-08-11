"""
predict.py
----------
Charge le meilleur modèle sauvegardé et prédit le churn pour un client donné.
Utilisé par l'application Streamlit.
"""

import os
import joblib
import pandas as pd
import numpy as np

MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
MODEL_PATH  = os.path.join(MODELS_DIR, 'best_model.joblib')

# Colonnes dans le bon ordre (après encodage One-Hot)
FEATURE_COLS = None   # chargé dynamiquement depuis le dataset traité


def load_model_and_scaler():
    """Charge le modèle et le scaler depuis le dossier models/."""
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def preprocess_single(client_dict, scaler):
    """
    Transforme un dictionnaire client (saisie manuelle) en DataFrame
    prêt pour la prédiction.

    Paramètres attendus dans client_dict :
        age, anciennete_mois, revenu_estime_gnf, recharge_mensuelle_moy_gnf,
        minutes_jour, minutes_nuit, minutes_internationales, donnees_mo,
        nombre_sms, appels_service_client, pannes_signalees_30j,
        nombre_reclamations, retard_paiement_jours,
        sexe (Homme/Femme), type_abonnement (Prépayé/Postpayé),
        forfait_international (Oui/Non), messagerie_vocale (Oui/Non),
        region (str), moyen_paiement (str)
    """
    d = client_dict.copy()

    # Calcul du ratio data/voix
    total_min = max(1, d['minutes_jour'] + d['minutes_nuit']
                    + d['minutes_internationales'])
    d['ratio_data_voix'] = round(d['donnees_mo'] / total_min, 2)

    # Encodage binaire
    d['sexe']                  = 1 if d['sexe'] == 'Homme' else 0
    d['type_abonnement']       = 1 if d['type_abonnement'] == 'Prépayé' else 0
    d['forfait_international'] = 1 if d['forfait_international'] == 'Oui' else 0
    d['messagerie_vocale']     = 1 if d['messagerie_vocale'] == 'Oui' else 0

    # One-Hot région
    regions = ['Boké', 'Conakry', 'Faranah', 'Kankan', 'Kindia', 'Labé',
               'Mamou', "N'Zérékoré"]
    for r in regions[1:]:   # drop_first=True → Boké est la référence
        d[f'region_{r}'] = 1 if d['region'] == r else 0
    del d['region']

    # One-Hot moyen de paiement
    moyens = ['Carte bancaire', 'Espèces', 'Orange Money', 'Virement bancaire']
    for m in moyens[1:]:    # drop_first=True → Carte bancaire est référence
        d[f'moyen_paiement_{m}'] = 1 if d['moyen_paiement'] == m else 0
    del d['moyen_paiement']

    df = pd.DataFrame([d])

    # Normalisation des colonnes numériques
    numeric_cols = ['age', 'revenu_estime_gnf', 'anciennete_mois',
                    'recharge_mensuelle_moy_gnf', 'minutes_jour',
                    'minutes_nuit', 'minutes_internationales',
                    'donnees_mo', 'ratio_data_voix', 'nombre_sms',
                    'appels_service_client', 'pannes_signalees_30j',
                    'nombre_reclamations', 'retard_paiement_jours']
    existing_num = [c for c in numeric_cols if c in df.columns]
    df[existing_num] = scaler.transform(df[existing_num])

    return df


def predict_churn(client_dict):
    """
    Prédit la probabilité de churn pour un client.

    Retourne :
        dict avec 'churn' (bool), 'probabilite' (float 0-1),
        'segment_risque' (str), 'recommandation' (str)
    """
    model, scaler = load_model_and_scaler()
    df = preprocess_single(client_dict, scaler)

    # Aligner les colonnes avec le modèle
    model_features = model.get_booster().feature_names if hasattr(model, 'get_booster') \
                     else list(df.columns)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    proba    = model.predict_proba(df)[0][1]
    is_churn = proba >= 0.5

    # Segmentation du risque
    if proba < 0.40:
        segment   = "Faible risque"
        action    = "Aucune action immédiate nécessaire. Surveillance mensuelle."
    elif proba < 0.60:
        segment   = "Risque modéré"
        action    = "Envoyer un SMS avec offre data bonus (+500 Mo gratuits)."
    elif proba < 0.80:
        segment   = "Risque élevé"
        action    = "SMS + notification app + offre forfait réduit 15%."
    else:
        segment   = "Risque très élevé"
        action    = "Appel conseiller dans les 48h + remise personnalisée + offre VIP."

    return {
        'churn':          is_churn,
        'probabilite':    round(float(proba), 4),
        'segment_risque': segment,
        'recommandation': action,
    }


if __name__ == '__main__':
    # Test rapide
    exemple = {
        'age': 32, 'anciennete_mois': 4,
        'revenu_estime_gnf': 450000,
        'recharge_mensuelle_moy_gnf': 30000,
        'minutes_jour': 50, 'minutes_nuit': 20,
        'minutes_internationales': 0,
        'donnees_mo': 800, 'nombre_sms': 5,
        'appels_service_client': 4,
        'pannes_signalees_30j': 2,
        'nombre_reclamations': 6,
        'retard_paiement_jours': 0,
        'sexe': 'Homme',
        'type_abonnement': 'Prépayé',
        'forfait_international': 'Non',
        'messagerie_vocale': 'Oui',
        'region': 'Conakry',
        'moyen_paiement': 'Orange Money',
    }
    result = predict_churn(exemple)
    print("\n=== Résultat de prédiction ===")
    for k, v in result.items():
        print(f"  {k:<20} : {v}")
