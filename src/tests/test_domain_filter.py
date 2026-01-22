"""
Test du filtrage par domaines d'expertise
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from matching import ProviderMatcher
from utils import load_providers, load_needs
from config import PRESTATAIRES_CSV, BESOINS_ENRICHIS_CSV


def test_domain_filter():
    """
    Teste le filtrage par domaines d'expertise
    """
    print("="*80)
    print("TEST DU FILTRAGE PAR DOMAINES D'EXPERTISE")
    print("="*80)
    
    # 1. Initialiser le matcher
    matcher = ProviderMatcher()
    providers_df = load_providers(PRESTATAIRES_CSV)
    matcher.load_providers(providers_df)
    matcher.encode_providers(show_progress=False)
    
    print(f"\n>> {len(providers_df)} prestataires chargés\n")
    
    # 2. Charger un besoin
    needs_df = load_needs(BESOINS_ENRICHIS_CSV)
    
    # Test 1: Garde d'enfant
    print("="*80)
    print("TEST 1: Garde d'enfant à Paris")
    print("="*80)
    
    besoin = needs_df[needs_df['Sous_Categorie'] == "Garde d'enfant"].iloc[0]
    print(f"Besoin: {besoin['Message_Utilisateur'][:80]}...")
    
    print("\n▸ SANS filtre domaines:")
    results_no_filter = matcher.match_need_row(besoin, top_k=5, apply_domain_filter=False)
    print(f"  {len(results_no_filter)} résultats")
    if len(results_no_filter) > 0:
        print(f"  Meilleur score: {results_no_filter.iloc[0]['similarity_score']:.4f}")
        print(f"  Top 3:")
        for i, row in results_no_filter.head(3).iterrows():
            print(f"    {i+1}. {row['Nom_Entreprise']:30s} | {row['Domaines_Expertise']:40s} | Score: {row['similarity_score']:.4f}")
    
    print("\n▸ AVEC filtre domaines:")
    results_with_filter = matcher.match_need_row(besoin, top_k=5, apply_domain_filter=True)
    print(f"  {len(results_with_filter)} résultats")
    if len(results_with_filter) > 0:
        print(f"  Meilleur score: {results_with_filter.iloc[0]['similarity_score']:.4f}")
        print(f"  Top 3:")
        for i, row in results_with_filter.head(3).iterrows():
            print(f"    {i+1}. {row['Nom_Entreprise']:30s} | {row['Domaines_Expertise']:40s} | Score: {row['similarity_score']:.4f}")
    
    # Test 2: Prêt immobilier
    print("\n" + "="*80)
    print("TEST 2: Prêt immobilier à Marseille")
    print("="*80)
    
    besoin2 = needs_df[needs_df['Sous_Categorie'] == "Prêt immobilier"].iloc[0]
    print(f"Besoin: {besoin2['Message_Utilisateur'][:80]}...")
    
    print("\n▸ SANS filtre domaines:")
    results_no_filter2 = matcher.match_need_row(besoin2, top_k=5, apply_domain_filter=False)
    print(f"  {len(results_no_filter2)} résultats")
    if len(results_no_filter2) > 0:
        print(f"  Meilleur score: {results_no_filter2.iloc[0]['similarity_score']:.4f}")
        print(f"  Top 3:")
        for i, row in results_no_filter2.head(3).iterrows():
            print(f"    {i+1}. {row['Nom_Entreprise']:30s} | {row['Domaines_Expertise']:40s} | Score: {row['similarity_score']:.4f}")
    
    print("\n▸ AVEC filtre domaines:")
    results_with_filter2 = matcher.match_need_row(besoin2, top_k=5, apply_domain_filter=True)
    print(f"  {len(results_with_filter2)} résultats")
    if len(results_with_filter2) > 0:
        print(f"  Meilleur score: {results_with_filter2.iloc[0]['similarity_score']:.4f}")
        print(f"  Top 3:")
        for i, row in results_with_filter2.head(3).iterrows():
            print(f"    {i+1}. {row['Nom_Entreprise']:30s} | {row['Domaines_Expertise']:40s} | Score: {row['similarity_score']:.4f}")
    
    # Test 3: Réparation urgente
    print("\n" + "="*80)
    print("TEST 3: Réparation urgente véhicule à Bourges")
    print("="*80)
    
    besoin3 = needs_df[needs_df['Sous_Categorie'] == "Réparation urgente"].iloc[0]
    print(f"Besoin: {besoin3['Message_Utilisateur'][:80]}...")
    
    print("\n▸ SANS filtre domaines:")
    results_no_filter3 = matcher.match_need_row(besoin3, top_k=5, apply_domain_filter=False)
    print(f"  {len(results_no_filter3)} résultats")
    if len(results_no_filter3) > 0:
        print(f"  Meilleur score: {results_no_filter3.iloc[0]['similarity_score']:.4f}")
        print(f"  Top 3:")
        for i, row in results_no_filter3.head(3).iterrows():
            print(f"    {i+1}. {row['Nom_Entreprise']:30s} | {row['Domaines_Expertise']:40s} | Score: {row['similarity_score']:.4f}")
    
    print("\n▸ AVEC filtre domaines:")
    results_with_filter3 = matcher.match_need_row(besoin3, top_k=5, apply_domain_filter=True)
    print(f"  {len(results_with_filter3)} résultats")
    if len(results_with_filter3) > 0:
        print(f"  Meilleur score: {results_with_filter3.iloc[0]['similarity_score']:.4f}")
        print(f"  Top 3:")
        for i, row in results_with_filter3.head(3).iterrows():
            print(f"    {i+1}. {row['Nom_Entreprise']:30s} | {row['Domaines_Expertise']:40s} | Score: {row['similarity_score']:.4f}")
    
    print("\n" + "="*80)
    print("✅ Tests terminés")
    print("="*80)


if __name__ == "__main__":
    test_domain_filter()
