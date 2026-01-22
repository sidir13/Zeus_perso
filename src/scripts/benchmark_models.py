"""
Benchmark de comparaison de différents modèles d'embeddings
pour le matching besoin-prestataires
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config import BESOINS_CSV, PRESTATAIRES_CSV, RAW_DATA_DIR, PROCESSED_DATA_DIR
from matching import ProviderMatcher
from utils import load_providers, load_needs
from utils.evaluation_metrics import evaluate_model_results, compute_matching_score


# Liste des modèles à tester
MODELS_TO_TEST = [
    {
        'name': 'CamemBERT-Large',
        'model_id': 'dangvantuan/sentence-camembert-large',
        'description': 'Modèle français spécialisé (1024 dim)'
    },
    {
        'name': 'Multilingual-MiniLM',
        'model_id': 'paraphrase-multilingual-MiniLM-L12-v2',
        'description': 'Modèle multilingue rapide (384 dim)'
    },
    {
        'name': 'Multilingual-MPNet',
        'model_id': 'paraphrase-multilingual-mpnet-base-v2',
        'description': 'Modèle multilingue précis (768 dim)'
    },
    {
        'name': 'Distiluse-Multilingual',
        'model_id': 'distiluse-base-multilingual-cased-v2',
        'description': 'Modèle multilingue équilibré (512 dim)'
    },
]


def load_ground_truth():
    """Charge le fichier ground truth"""
    gt_path = RAW_DATA_DIR / 'ground_truth.csv'
    df = pd.read_csv(gt_path, sep=';', encoding='utf-8')
    print(f">> Ground truth chargé: {len(df)} associations")
    return df


def run_matching_for_model(model_info, needs_df, providers_df, top_k=5):
    """
    Exécute le matching pour un modèle donné
    
    Returns:
        DataFrame: Résultats avec [besoin_id, prestataire_id, similarity_score, rank]
    """
    print(f"\n{'='*80}")
    print(f"Test du modèle: {model_info['name']}")
    print(f"Description: {model_info['description']}")
    print(f"{'='*80}")
    
    # Initialiser le matcher
    start_time = time.time()
    matcher = ProviderMatcher(model_name=model_info['model_id'])
    matcher.load_providers(providers_df)
    
    # Encoder les prestataires
    print("Encodage des prestataires...")
    matcher.encode_providers(show_progress=True)
    encoding_time = time.time() - start_time
    
    # Faire le matching pour chaque besoin
    print(f"\nMatching de {len(needs_df)} besoins...")
    all_results = []
    
    matching_start = time.time()
    for idx, need_row in tqdm(needs_df.iterrows(), total=len(needs_df), desc="Matching"):
        results = matcher.match_need_row(need_row, top_k=top_k, threshold=0.0)
        
        # Ajouter le besoin_id et le rank
        for rank, (_, provider_row) in enumerate(results.iterrows(), 1):
            all_results.append({
                'besoin_id': idx,
                'prestataire_id': provider_row.name,
                'similarity_score': provider_row['similarity_score'],
                'rank': rank,
                'nom_entreprise': provider_row['Nom_Entreprise']
            })
    
    matching_time = time.time() - matching_start
    total_time = time.time() - start_time
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n>> Temps d'encodage: {encoding_time:.2f}s")
    print(f">> Temps de matching: {matching_time:.2f}s")
    print(f">> Temps total: {total_time:.2f}s")
    
    return results_df, {
        'encoding_time': encoding_time,
        'matching_time': matching_time,
        'total_time': total_time
    }


def evaluate_and_compare_models():
    """Fonction principale de benchmark"""
    
    print("\n" + "="*80)
    print("BENCHMARK DE MODELES D'EMBEDDINGS POUR LE MATCHING")
    print("="*80)
    
    # Charger les données
    print("\nChargement des données...")
    needs_df = load_needs(BESOINS_CSV)
    providers_df = load_providers(PRESTATAIRES_CSV)
    ground_truth_df = load_ground_truth()
    
    # Résultats de tous les modèles
    all_model_results = []
    
    # Tester chaque modèle
    for model_info in MODELS_TO_TEST:
        try:
            # Exécuter le matching
            predictions_df, timing_info = run_matching_for_model(
                model_info, 
                needs_df, 
                providers_df,
                top_k=5
            )
            
            # Évaluer les résultats
            print("\nÉvaluation des métriques...")
            metrics = evaluate_model_results(predictions_df, ground_truth_df, k_values=[1, 3, 5])
            
            # Calculer le score global
            global_score = compute_matching_score(metrics)
            
            # Stocker les résultats
            result = {
                'model_name': model_info['name'],
                'model_id': model_info['model_id'],
                'description': model_info['description'],
                'global_score': global_score,
                **metrics,
                **timing_info
            }
            
            all_model_results.append(result)
            
            # Afficher les résultats
            print(f"\n>> RESULTATS POUR {model_info['name']}:")
            print(f"   Score Global: {global_score:.2f}/100")
            print(f"   Precision@1: {metrics['precision@1']:.3f}")
            print(f"   Precision@3: {metrics['precision@3']:.3f}")
            print(f"   Recall@5: {metrics['recall@5']:.3f}")
            print(f"   MAP: {metrics['MAP']:.3f}")
            print(f"   MRR: {metrics['MRR']:.3f}")
            
        except Exception as e:
            print(f"\n!! ERREUR avec le modèle {model_info['name']}: {e}")
            continue
    
    # Créer le DataFrame de résultats
    results_df = pd.DataFrame(all_model_results)
    
    # Sauvegarder
    output_path = PROCESSED_DATA_DIR / 'model_benchmark_results.csv'
    results_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
    print(f"\n>> Résultats sauvegardés dans {output_path}")
    
    # Générer les visualisations
    generate_comparison_charts(results_df)
    
    # Afficher le classement final
    display_final_ranking(results_df)
    
    return results_df


def generate_comparison_charts(results_df):
    """Génère des graphiques de comparaison"""
    
    print("\nGénération des graphiques de comparaison...")
    
    # 1. Graphique des scores globaux
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
    results_sorted = results_df.sort_values('global_score', ascending=False)
    
    plt.barh(results_sorted['model_name'], results_sorted['global_score'], color=colors)
    plt.xlabel('Score Global (/100)')
    plt.title('Comparaison des Scores Globaux par Modèle', fontsize=16, fontweight='bold')
    plt.xlim(0, 100)
    
    # Ajouter les valeurs sur les barres
    for i, (idx, row) in enumerate(results_sorted.iterrows()):
        plt.text(row['global_score'] + 1, i, f"{row['global_score']:.1f}", 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'benchmark_scores_globaux.png', dpi=300, bbox_inches='tight')
    print(">> Graphique sauvegardé: benchmark_scores_globaux.png")
    plt.show()
    plt.close()
    
    # 2. Radar chart des métriques
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    metrics_to_plot = ['precision@1', 'precision@3', 'recall@5', 'MAP', 'MRR']
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, row in results_df.iterrows():
        values = [row[m] for m in metrics_to_plot]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1)
    ax.set_title('Comparaison des Métriques par Modèle', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'benchmark_radar_metriques.png', dpi=300, bbox_inches='tight')
    print(">> Graphique sauvegardé: benchmark_radar_metriques.png")
    plt.show()
    plt.close()
    
    # 3. Comparaison des temps d'exécution
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['encoding_time'], width, label='Temps encodage', color='#e74c3c')
    plt.bar(x + width/2, results_df['matching_time'], width, label='Temps matching', color='#3498db')
    
    plt.xlabel('Modèles')
    plt.ylabel('Temps (secondes)')
    plt.title('Comparaison des Temps d\'Exécution', fontsize=16, fontweight='bold')
    plt.xticks(x, results_df['model_name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'benchmark_temps_execution.png', dpi=300, bbox_inches='tight')
    print(">> Graphique sauvegardé: benchmark_temps_execution.png")
    plt.show()
    plt.close()


def display_final_ranking(results_df):
    """Affiche le classement final des modèles"""
    
    print("\n" + "="*80)
    print("CLASSEMENT FINAL DES MODELES")
    print("="*80)
    
    results_sorted = results_df.sort_values('global_score', ascending=False)
    
    for rank, (idx, row) in enumerate(results_sorted.iterrows(), 1):
        medal = "  " if rank > 3 else ["", "", ""][rank-1]
        print(f"\n{medal} #{rank} - {row['model_name']}")
        print(f"   Score Global: {row['global_score']:.2f}/100")
        print(f"   Precision@1: {row['precision@1']:.3f} | Precision@3: {row['precision@3']:.3f}")
        print(f"   Recall@5: {row['recall@5']:.3f} | MAP: {row['MAP']:.3f} | MRR: {row['MRR']:.3f}")
        print(f"   Temps total: {row['total_time']:.2f}s")
        print(f"   Description: {row['description']}")
        print("-" * 80)
    
    # Recommandation
    best_model = results_sorted.iloc[0]
    print(f"\nMODELE RECOMMANDE: {best_model['model_name']}")
    print(f"Score: {best_model['global_score']:.2f}/100")
    print(f"Temps: {best_model['total_time']:.2f}s")


if __name__ == "__main__":
    results = evaluate_and_compare_models()
