"""
Métriques d'évaluation pour le matching
"""

import numpy as np
import pandas as pd


def calculate_precision_at_k(predicted_ids, relevant_ids, k):
    """
    Calcule la précision@k: proportion de résultats pertinents parmi les k premiers
    
    Args:
        predicted_ids: Liste ordonnée des IDs prédits
        relevant_ids: Liste des IDs pertinents (ground truth)
        k: Nombre de résultats à considérer
    
    Returns:
        float: Précision@k entre 0 et 1
    """
    if k == 0 or len(predicted_ids) == 0:
        return 0.0
    
    # Prendre les k premiers
    top_k = predicted_ids[:k]
    
    # Compter combien sont pertinents
    relevant_in_top_k = len(set(top_k) & set(relevant_ids))
    
    return relevant_in_top_k / k


def calculate_recall_at_k(predicted_ids, relevant_ids, k):
    """
    Calcule le recall@k: proportion de résultats pertinents trouvés parmi tous les pertinents
    
    Args:
        predicted_ids: Liste ordonnée des IDs prédits
        relevant_ids: Liste des IDs pertinents
        k: Nombre de résultats à considérer
    
    Returns:
        float: Recall@k entre 0 et 1
    """
    if len(relevant_ids) == 0:
        return 0.0
    
    # Prendre les k premiers
    top_k = predicted_ids[:k]
    
    # Compter combien de pertinents ont été trouvés
    relevant_found = len(set(top_k) & set(relevant_ids))
    
    return relevant_found / len(relevant_ids)


def calculate_average_precision(predicted_ids, relevant_ids, k=None):
    """
    Calcule l'Average Precision (AP)
    
    Args:
        predicted_ids: Liste ordonnée des IDs prédits
        relevant_ids: Liste des IDs pertinents
        k: Limite optionnelle (None = tous les résultats)
    
    Returns:
        float: Average Precision entre 0 et 1
    """
    if len(relevant_ids) == 0:
        return 0.0
    
    if k is not None:
        predicted_ids = predicted_ids[:k]
    
    score = 0.0
    num_relevant = 0
    
    for i, pred_id in enumerate(predicted_ids, 1):
        if pred_id in relevant_ids:
            num_relevant += 1
            precision_at_i = num_relevant / i
            score += precision_at_i
    
    if num_relevant == 0:
        return 0.0
    
    return score / len(relevant_ids)


def calculate_mean_average_precision(all_predictions, all_relevant, k=None):
    """
    Calcule le Mean Average Precision (MAP) sur plusieurs requêtes
    
    Args:
        all_predictions: Liste de listes (prédictions pour chaque requête)
        all_relevant: Liste de listes (pertinents pour chaque requête)
        k: Limite optionnelle
    
    Returns:
        float: MAP entre 0 et 1
    """
    if len(all_predictions) == 0:
        return 0.0
    
    aps = []
    for preds, rels in zip(all_predictions, all_relevant):
        ap = calculate_average_precision(preds, rels, k)
        aps.append(ap)
    
    return np.mean(aps)


def calculate_mrr(predicted_ids, relevant_ids):
    """
    Calcule le Mean Reciprocal Rank (MRR)
    
    Args:
        predicted_ids: Liste ordonnée des IDs prédits
        relevant_ids: Liste des IDs pertinents
    
    Returns:
        float: Reciprocal rank (1/rang du premier pertinent)
    """
    for i, pred_id in enumerate(predicted_ids, 1):
        if pred_id in relevant_ids:
            return 1.0 / i
    return 0.0


def calculate_ndcg_at_k(predicted_ids, relevant_ids, k):
    """
    Calcule le Normalized Discounted Cumulative Gain (NDCG@k)
    
    Args:
        predicted_ids: Liste ordonnée des IDs prédits
        relevant_ids: Liste des IDs pertinents
        k: Nombre de résultats à considérer
    
    Returns:
        float: NDCG@k entre 0 et 1
    """
    if len(relevant_ids) == 0:
        return 0.0
    
    # DCG
    dcg = 0.0
    for i, pred_id in enumerate(predicted_ids[:k], 1):
        if pred_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 1)
    
    # IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_model_results(predictions_df, ground_truth_df, k_values=[1, 3, 5]):
    """
    Évalue les résultats d'un modèle par rapport au ground truth
    
    Args:
        predictions_df: DataFrame avec colonnes [besoin_id, prestataire_id, rank/score]
        ground_truth_df: DataFrame avec colonnes [besoin_id, prestataire_id, pertinence]
        k_values: Liste des valeurs de k pour les métriques @k
    
    Returns:
        dict: Dictionnaire contenant toutes les métriques
    """
    metrics = {}
    
    # Grouper par besoin
    all_predictions = []
    all_relevant = []
    
    for besoin_id in ground_truth_df['besoin_id'].unique():
        # Prédictions pour ce besoin (triées par score/rank)
        preds = predictions_df[predictions_df['besoin_id'] == besoin_id]
        pred_ids = preds['prestataire_id'].tolist()
        
        # Vérité terrain pour ce besoin
        gt = ground_truth_df[
            (ground_truth_df['besoin_id'] == besoin_id) & 
            (ground_truth_df['pertinence'] == 1)
        ]
        relevant_ids = gt['prestataire_id'].tolist()
        
        all_predictions.append(pred_ids)
        all_relevant.append(relevant_ids)
    
    # Calculer les métriques pour chaque k
    for k in k_values:
        precisions = [calculate_precision_at_k(p, r, k) 
                     for p, r in zip(all_predictions, all_relevant)]
        recalls = [calculate_recall_at_k(p, r, k) 
                  for p, r in zip(all_predictions, all_relevant)]
        ndcgs = [calculate_ndcg_at_k(p, r, k) 
                for p, r in zip(all_predictions, all_relevant)]
        
        metrics[f'precision@{k}'] = np.mean(precisions)
        metrics[f'recall@{k}'] = np.mean(recalls)
        metrics[f'ndcg@{k}'] = np.mean(ndcgs)
    
    # MAP
    metrics['MAP'] = calculate_mean_average_precision(all_predictions, all_relevant)
    metrics['MAP@5'] = calculate_mean_average_precision(all_predictions, all_relevant, k=5)
    
    # MRR
    mrrs = [calculate_mrr(p, r) for p, r in zip(all_predictions, all_relevant)]
    metrics['MRR'] = np.mean(mrrs)
    
    return metrics


def compute_matching_score(metrics, weights=None):
    """
    Calcule un score global de matching basé sur les métriques
    
    Args:
        metrics: Dict contenant les métriques
        weights: Dict optionnel des poids pour chaque métrique
    
    Returns:
        float: Score global entre 0 et 100
    """
    if weights is None:
        # Poids par défaut
        weights = {
            'precision@1': 0.15,
            'precision@3': 0.20,
            'precision@5': 0.15,
            'recall@5': 0.15,
            'MAP': 0.20,
            'MRR': 0.15
        }
    
    score = 0.0
    total_weight = 0.0
    
    for metric_name, weight in weights.items():
        if metric_name in metrics:
            score += metrics[metric_name] * weight
            total_weight += weight
    
    # Normaliser et convertir en pourcentage
    if total_weight > 0:
        score = (score / total_weight) * 100
    
    return score
