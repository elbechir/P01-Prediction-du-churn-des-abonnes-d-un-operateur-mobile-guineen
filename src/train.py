"""
train.py
--------
Entraîne et compare 3 algorithmes (Régression Logistique, Random Forest, XGBoost).
Sauvegarde le meilleur modèle en .joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from preprocessing import run_pipeline

# ── Chemins ──────────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_PATH = os.path.join(MODELS_DIR, 'resultats_modeles.json')

# ── Modèles à comparer ───────────────────────────────────────────────────────
MODELS = {
    'Regression_Logistique': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'),
    'Random_Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        class_weight='balanced', n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=3,   # gère le déséquilibre
        random_state=42, eval_metric='logloss', verbosity=0),
    'XGBoost_SMOTE': None,    # entraîné sur données SMOTE, défini plus bas
}


def evaluate_model(model, X_test, y_test, name):
    """Calcule et affiche toutes les métriques."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred),  4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred),    4),
        'f1':        round(f1_score(y_test, y_pred),        4),
        'roc_auc':   round(roc_auc_score(y_test, y_proba),  4),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12} : {v}")
    print(f"\n  Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    print(f"\n  ⚠  Faux Négatifs (churners manqués) : {cm[1,0]}")
    print(f"  Coût estimé : {cm[1,0]} × 250 000 GNF = "
          f"{cm[1,0]*250000:,} GNF")

    return metrics


def get_feature_importance(model, feature_names, model_name, top_n=10):
    """Retourne les top N variables explicatives."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return {}

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top = fi.head(top_n)
    print(f"\n  Top {top_n} variables ({model_name}) :")
    for var, imp in top.items():
        print(f"    {var:<35} {imp:.4f}")
    return top.to_dict()


def train_and_evaluate():
    """Pipeline complet : entraînement de tous les modèles + comparaison."""

    # ── Données sans SMOTE ──────────────────────────────────────────────────
    print("\n[1/2] Pipeline SANS SMOTE …")
    X_train, X_test, y_train, y_test, _ = run_pipeline(apply_smote_flag=False)

    results = {}
    trained_models = {}

    for name, model in MODELS.items():
        if name == 'XGBoost_SMOTE':
            continue   # traité séparément
        print(f"\n  Entraînement : {name} …")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test, name)
        results[name]['features'] = get_feature_importance(
            model, X_train.columns.tolist(), name)
        trained_models[name] = model

    # ── Données avec SMOTE ──────────────────────────────────────────────────
    print("\n[2/2] Pipeline AVEC SMOTE …")
    X_train_sm, _, y_train_sm, _, _ = run_pipeline(apply_smote_flag=True)

    xgb_smote = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', verbosity=0)

    print("\n  Entraînement : XGBoost_SMOTE …")
    xgb_smote.fit(X_train_sm, y_train_sm)
    results['XGBoost_SMOTE'] = evaluate_model(xgb_smote, X_test, y_test,
                                               'XGBoost_SMOTE')
    results['XGBoost_SMOTE']['features'] = get_feature_importance(
        xgb_smote, X_train.columns.tolist(), 'XGBoost_SMOTE')
    trained_models['XGBoost_SMOTE'] = xgb_smote

    # ── Sélectionner le meilleur modèle (F1) ────────────────────────────────
    best_name = max(results, key=lambda k: results[k]['f1'])
    best_model = trained_models[best_name]
    print(f"\n✅ Meilleur modèle : {best_name}  (F1 = {results[best_name]['f1']})")

    # ── Sauvegarder ─────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in trained_models.items():
        path = os.path.join(MODELS_DIR, f'{name}.joblib')
        joblib.dump(model, path)
        print(f"  Modèle sauvegardé → {path}")

    best_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(best_model, best_path)
    print(f"  Meilleur modèle → {best_path}")

    # Sauvegarder les métriques JSON (pour le README)
    clean_results = {k: {m: v for m, v in r.items() if m != 'features'}
                     for k, r in results.items()}
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Résultats JSON → {RESULTS_PATH}")

    # ── Tableau comparatif final ─────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  TABLEAU COMPARATIF FINAL")
    print("="*70)
    print(f"{'Modèle':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-"*70)
    for name, r in results.items():
        star = " ⭐" if name == best_name else ""
        print(f"{name:<25} {r['accuracy']:>6} {r['precision']:>6} "
              f"{r['recall']:>6} {r['f1']:>6} {r['roc_auc']:>6}{star}")

    return results, trained_models, best_name


if __name__ == '__main__':
    train_and_evaluate()
# Meilleur modele : Regression Logistique - F1=0.405 - Rappel=0.577