"""
Fonctions d'affichage des résultats de matching
"""

import pandas as pd


def display_matches(results, max_display=5):
    """
    Affiche les résultats de matching de manière lisible
    
    Args:
        results: DataFrame contenant les résultats avec colonne 'similarity_score'
        max_display: Nombre maximum de résultats à afficher
    """
    print("\n" + "="*80)
    print("MEILLEURS PRESTATAIRES TROUVES")
    print("="*80)
    
    if len(results) == 0:
        print(">> Aucun prestataire ne correspond aux critères")
        return
    
    # Limiter l'affichage
    display_results = results.head(max_display)
    
    for idx, (_, row) in enumerate(display_results.iterrows(), 1):
        score_pct = row['similarity_score'] * 100
        
        print(f"\n#{idx} - Score de similarité: {score_pct:.1f}%")
        print(f"   Entreprise: {row['Nom_Entreprise']}")
        
        if pd.notna(row.get('Ville')):
            print(f"   Ville: {row['Ville']}")
        
        if pd.notna(row.get('Domaines_Expertise')):
            print(f"   Expertise: {row['Domaines_Expertise']}")
        
        if pd.notna(row.get('Disponibilite')):
            print(f"   Disponibilité: {row['Disponibilite']}")
        
        if pd.notna(row.get('Description_Service')):
            desc = str(row['Description_Service'])
            if len(desc) > 150:
                desc = desc[:150] + "..."
            print(f"   Services: {desc}")
        
        print("-" * 80)
    
    if len(results) > max_display:
        print(f"\n... et {len(results) - max_display} autres résultats")


def display_match_summary(results):
    """
    Affiche un résumé des résultats de matching
    
    Args:
        results: DataFrame contenant les résultats avec colonne 'similarity_score'
    """
    if len(results) == 0:
        print("\n>> Aucun match trouvé")
        return
    
    print("\n" + "="*80)
    print("RESUME DES RESULTATS")
    print("="*80)
    
    print(f"\nNombre de matches trouvés: {len(results)}")
    print(f"Score moyen: {results['similarity_score'].mean()*100:.1f}%")
    print(f"Meilleur score: {results['similarity_score'].max()*100:.1f}%")
    print(f"Score le plus bas: {results['similarity_score'].min()*100:.1f}%")
    
    # Répartition par disponibilité
    if 'Disponibilite' in results.columns:
        print("\nRépartition par disponibilité:")
        avail_counts = results['Disponibilite'].value_counts()
        for avail, count in avail_counts.items():
            print(f"  - {avail}: {count}")


def display_request_info(request_text):
    """
    Affiche les informations sur la requête client
    
    Args:
        request_text: Texte de la requête formaté
    """
    print("\n" + "="*80)
    print("REQUETE CLIENT")
    print("="*80)
    print(f"\n{request_text}")


def display_detailed_match(provider_row):
    """
    Affiche les détails complets d'un prestataire
    
    Args:
        provider_row: pandas.Series contenant les données du prestataire
    """
    print("\n" + "="*80)
    print("DETAILS DU PRESTATAIRE")
    print("="*80)
    
    if 'Nom_Entreprise' in provider_row:
        print(f"\nEntreprise: {provider_row['Nom_Entreprise']}")
    
    if 'similarity_score' in provider_row:
        score_pct = provider_row['similarity_score'] * 100
        print(f"Score de similarité: {score_pct:.1f}%")
    
    if 'Domaines_Expertise' in provider_row:
        print(f"\nDomaines d'expertise:")
        domains = str(provider_row['Domaines_Expertise']).split(',')
        for domain in domains:
            print(f"  - {domain.strip()}")
    
    if 'Disponibilite' in provider_row:
        print(f"\nDisponibilité: {provider_row['Disponibilite']}")
    
    if 'Description_Service' in provider_row:
        print(f"\nDescription des services:")
        print(f"  {provider_row['Description_Service']}")


def create_results_table(results):
    """
    Crée un tableau récapitulatif des résultats pour affichage
    
    Args:
        results: DataFrame contenant les résultats
    
    Returns:
        pandas.DataFrame: Tableau formaté pour l'affichage
    """
    if len(results) == 0:
        return pd.DataFrame()
    
    display_df = results.copy()
    
    # Formater le score en pourcentage
    if 'similarity_score' in display_df.columns:
        display_df['Score (%)'] = (display_df['similarity_score'] * 100).round(1)
        display_df = display_df.drop('similarity_score', axis=1)
    
    # Tronquer les descriptions longues
    if 'Description_Service' in display_df.columns:
        display_df['Description_Service'] = display_df['Description_Service'].apply(
            lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x)
        )
    
    return display_df
