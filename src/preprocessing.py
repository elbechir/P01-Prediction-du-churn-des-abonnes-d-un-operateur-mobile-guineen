"""
preprocessing.py
----------------
Nettoyage, encodage et préparation du dataset churn TélécomGuinée.
Utilisé par les notebooks et le script d'entraînement.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Chemins ──────────────────────────────────────────────────────────────────
RAW_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',
                         'guinee_telecom_churn_FR.csv')
PROC_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed',
                         'churn_processed.csv')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models',
                            'scaler.joblib')

# ── Variables ─────────────────────────────────────────────────────────────────
COLS_TO_DROP   = ['id_client', 'retour_client']
TARGET         = 'resiliation'
COLS_BINARY    = ['forfait_international', 'messagerie_vocale',
                  'type_abonnement', 'sexe']
COLS_ONEHOT    = ['region', 'moyen_paiement']
COLS_NUMERIC   = ['age', 'revenu_estime_gnf', 'anciennete_mois',
                  'recharge_mensuelle_moy_gnf', 'minutes_jour',
                  'minutes_nuit', 'minutes_internationales',
                  'donnees_mo', 'ratio_data_voix', 'nombre_sms',
                  'appels_service_client', 'pannes_signalees_30j',
                  'nombre_reclamations', 'retard_paiement_jours']


def load_data(path=RAW_PATH):
    """Charge le CSV brut."""
    df = pd.read_csv(path)
    print(f"[load] {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


def check_quality(df):
    """Affiche un rapport qualité rapide."""
    print("\n=== Rapport qualité ===")
    print(f"Doublons : {df.duplicated().sum()}")
    missing = df.isnull().sum()
    print(f"Valeurs manquantes :\n{missing[missing > 0]}")
    print(f"\nDistribution cible :\n{df[TARGET].value_counts(normalize=True).round(3)}")


def encode_features(df):
    """
    Encodage des variables catégorielles :
    - Binary (Oui/Non, Homme/Femme, Prépayé/Postpayé) → 0/1 avec LabelEncoder
    - Nominales (region, moyen_paiement) → One-Hot
    """
    df = df.copy()

    # Supprimer colonnes inutiles
    df = df.drop(columns=COLS_TO_DROP, errors='ignore')

    # Cible
    df[TARGET] = (df[TARGET] == 'Oui').astype(int)

    # Variables binaires
    le = LabelEncoder()
    for col in COLS_BINARY:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # One-Hot
    df = pd.get_dummies(df, columns=COLS_ONEHOT, drop_first=True)

    return df


def scale_features(df, fit=True, scaler=None):
    """Normalise les variables numériques (StandardScaler)."""
    numeric_cols = [c for c in COLS_NUMERIC if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        joblib.dump(scaler, SCALER_PATH)
        print(f"[scale] Scaler sauvegardé → {SCALER_PATH}")
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler


def split_data(df, test_size=0.2, random_state=42):
    """Sépare features et cible, puis train/test."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"[split] Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state=42):
    """Rééquilibrage par SMOTE sur l'ensemble d'entraînement uniquement."""
    print(f"[SMOTE] Avant : {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] Après : {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def run_pipeline(apply_smote_flag=True):
    """
    Pipeline complet : charge → encode → scale → split → (SMOTE).
    Retourne X_train, X_test, y_train, y_test et le scaler.
    """
    df = load_data()
    check_quality(df)
    df = encode_features(df)
    df, scaler = scale_features(df, fit=True)

    # Sauvegarder le dataset traité
    df.to_csv(PROC_PATH, index=False)
    print(f"[save] Dataset traité → {PROC_PATH}")

    X_train, X_test, y_train, y_test = split_data(df)

    if apply_smote_flag:
        X_train, y_train = apply_smote(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == '__main__':
    run_pipeline()
# Pipeline validé - version 1.0