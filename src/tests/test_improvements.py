"""
Test des 3 améliorations du matching
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from matching import ProviderMatcher
from utils import load_providers, load_needs
from config import PRESTATAIRES_CSV, BESOINS_ENRICHIS_CSV, MATCHING_CONFIG


def compare_improvements():
    """
    Compare les scores avant/après améliorations
    """
    print("="*80)
    print("TEST DES AMÉLIORATIONS DU MATCHING")
    print("="*80)
    
    # Initialiser le matcher
    matcher = ProviderMatcher(model_name=MATCHING_CONFIG['model_name'])
    providers_df = load_providers(PRESTATAIRES_CSV)
    matcher.load_providers(providers_df)
    
    print("\n>> Encodage des prestataires...")
    matcher.encode_providers(show_progress=False)
    
    # Charger les besoins
    needs_df = load_needs(BESOINS_ENRICHIS_CSV)
    
    # Test 1: Garde d'enfant URGENTE
    print("\n" + "="*80)
    print("TEST 1: Garde d'enfant URGENTE à Paris (impact_geo=2)")
    print("="*80)
    
    besoin1 = needs_df[needs_df['Sous_Categorie'] == "Garde d'enfant"].iloc[0]
    print(f"\nBesoin: {besoin1['Message_Utilisateur'][:100]}...")
    print(f"Urgence: {besoin1['Urgence_Deduite']}")
    print(f"Impact_Geo: {besoin1['Impact_Geo']}")
    
    results1 = matcher.match_need_row(besoin1, top_k=5, threshold=0.25)
    
    print(f"\n✓ {len(results1)} résultats trouvés")
    print("\nTop 5:")
    print(f"{'Rank':<6} {'Prestataire':<35} {'Domaines':<25} {'Dispo':<15} {'Score':<8}")
    print("-"*95)
    
    for i, (_, row) in enumerate(results1.iterrows(), 1):
        nom = row['Nom_Entreprise'][:33]
        domaines = row['Domaines_Expertise'].split(',')[0][:23]
        dispo = row['Disponibilite'][:13]
        score = row['similarity_score']
        
        # Indicateurs
        indicators = []
        if '24/7' in row['Disponibilite'] or 'urgence' in row['Disponibilite'].lower():
            indicators.append("⚡")
        if hasattr(row, 'urgency_boost') and row.get('urgency_boost', 1.0) > 1.0:
            indicators.append("+15%")
        if hasattr(row, 'specialization_factor'):
            if row['specialization_factor'] < 1.0:
                indicators.append(f"-{int((1-row['specialization_factor'])*100)}%")
        
        indicator_str = " ".join(indicators)
        
        print(f"{i:<6} {nom:<35} {domaines:<25} {dispo:<15} {score:.4f} {indicator_str}")
    
    # Test 2: Prêt immobilier (impact_geo=0, pas d'urgence)
    print("\n" + "="*80)
    print("TEST 2: Prêt immobilier à Marseille (impact_geo=0, service en ligne)")
    print("="*80)
    
    besoin2 = needs_df[needs_df['Sous_Categorie'] == "Prêt immobilier"].iloc[0]
    print(f"\nBesoin: {besoin2['Message_Utilisateur'][:100]}...")
    print(f"Urgence: {besoin2['Urgence_Deduite']}")
    print(f"Impact_Geo: {besoin2['Impact_Geo']}")
    
    results2 = matcher.match_need_row(besoin2, top_k=5, threshold=0.25)
    
    print(f"\n✓ {len(results2)} résultats trouvés")
    print("\nTop 5:")
    print(f"{'Rank':<6} {'Prestataire':<35} {'Domaines':<25} {'Score Base':<12} {'Score Final':<12}")
    print("-"*95)
    
    for i, (_, row) in enumerate(results2.iterrows(), 1):
        nom = row['Nom_Entreprise'][:33]
        domaines = row['Domaines_Expertise'].split(',')[0][:23]
        score_base = row.get('similarity_score_base', row['similarity_score'])
        score_final = row['similarity_score']
        
        print(f"{i:<6} {nom:<35} {domaines:<25} {score_base:.4f}       {score_final:.4f}")
    
    # Test 3: Location meublée (impact_geo=1, modéré)
    print("\n" + "="*80)
    print("TEST 3: Location meublée à Lyon (impact_geo=1, service local)")
    print("="*80)
    
    besoin3 = needs_df[needs_df['Sous_Categorie'] == "Location meublée"].iloc[0]
    print(f"\nBesoin: {besoin3['Message_Utilisateur'][:100]}...")
    print(f"Urgence: {besoin3['Urgence_Deduite']}")
    print(f"Impact_Geo: {besoin3['Impact_Geo']}")
    
    results3 = matcher.match_need_row(besoin3, top_k=5, threshold=0.25)
    
    print(f"\n✓ {len(results3)} résultats trouvés")
    print("\nTop 5:")
    print(f"{'Rank':<6} {'Prestataire':<35} {'Ville':<15} {'Geo Score':<12} {'Final Score':<12}")
    print("-"*85)
    
    for i, (_, row) in enumerate(results3.iterrows(), 1):
        nom = row['Nom_Entreprise'][:33]
        ville = row.get('Ville', 'N/A')[:13]
        geo_score = row.get('geo_score', 1.0)
        final_score = row['similarity_score']
        
        print(f"{i:<6} {nom:<35} {ville:<15} {geo_score:.4f}       {final_score:.4f}")
    
    # Analyse des pénalités généralistes
    print("\n" + "="*80)
    print("ANALYSE: Impact de la pénalité généralistes")
    print("="*80)
    
    all_providers = providers_df.copy()
    all_providers['nb_domaines'] = all_providers['Domaines_Expertise'].apply(
        lambda x: len([d.strip() for d in str(x).split(',') if d.strip()])
    )
    
    print("\nDistribution des prestataires par nombre de domaines:")
    dist = all_providers['nb_domaines'].value_counts().sort_index()
    for nb, count in dist.items():
        if nb <= 3:
            penalty = "0%"
        elif nb == 4:
            penalty = "-5%"
        elif nb == 5:
            penalty = "-10%"
        else:
            penalty = "-15%"
        print(f"  {nb} domaines: {count:2d} prestataires (pénalité: {penalty})")
    
    # Statistiques finales
    print("\n" + "="*80)
    print("RÉSUMÉ DES AMÉLIORATIONS APPLIQUÉES")
    print("="*80)
    
    print("\n✓ Amélioration #2: Boost urgence (+15% si 24/7 + IMMEDIATE)")
    print("✓ Amélioration #3: Pénalité généralistes (-5% à -15% selon nb domaines)")
    print("✓ Amélioration #4: Pondération adaptive (poids sémantique/geo selon impact_geo)")
    
    print("\nPoids par impact_geo:")
    print("  impact=0 (services en ligne): 100% sémantique, 0% geo")
    print("  impact=1 (services locaux):    65% sémantique, 35% geo")
    print("  impact=2 (urgences):           45% sémantique, 55% geo")


if __name__ == "__main__":
    compare_improvements()
