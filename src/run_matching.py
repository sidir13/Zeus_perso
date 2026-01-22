"""
Script principal pour exécuter le matching entre besoins et prestataires
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent))

from config import BESOINS_CSV, BESOINS_ENRICHIS_CSV, PRESTATAIRES_CSV, PROCESSED_DATA_DIR, MATCHING_CONFIG
from matching import ProviderMatcher
from utils import load_providers, save_matches, display_matches, display_request_info


def example_single_request():
    """Exemple: Trouver des prestataires pour une demande unique"""
    
    print("\n" + "="*80)
    print("EXEMPLE 1: MATCHING POUR UNE DEMANDE UNIQUE")
    print("="*80)
    
    # 1. Initialiser le matcher
    matcher = ProviderMatcher(model_name=MATCHING_CONFIG['model_name'])
    
    # 2. Charger les prestataires
    providers_df = load_providers(PRESTATAIRES_CSV)
    matcher.load_providers(providers_df)
    
    # 3. Générer les embeddings des prestataires (à faire UNE FOIS)
    matcher.encode_providers()
    
    # 4. Définir la demande client
    client_request = {
        'categorie': 'Besoins de dernière minute',
        'service': 'Garde d\'enfant',
        'ville': 'Paris',
        'description': 'Je pars en mission imprévue demain matin, besoin d\'une solution de garde pour mes 2 enfants (3 et 5 ans) dès 6h',
        'urgence': 'Immédiat'
    }
    
    # Afficher la requête
    from matching.text_processor import create_client_request_text
    request_text = create_client_request_text(client_request)
    display_request_info(request_text)
    
    # 5. Trouver les meilleurs matches
    results = matcher.find_matches(
        client_request,
        top_k=MATCHING_CONFIG['top_k'],
        threshold=MATCHING_CONFIG['threshold'],
        impact_geo=2  # Garde d'enfant = proximité critique
    )
    
    # 6. Afficher les résultats
    display_matches(results)
    
    # 7. Sauvegarder si besoin
    if len(results) > 0:
        output_path = PROCESSED_DATA_DIR / 'match_garde_enfant.csv'
        save_matches(results, output_path)
    
    return matcher, results


def example_from_csv():
    """Exemple: Matcher un besoin depuis le CSV besoins.csv"""
    
    print("\n" + "="*80)
    print("EXEMPLE 2: MATCHING DEPUIS LE CSV BESOINS")
    print("="*80)
    
    # 1. Initialiser le matcher et charger les données
    matcher = ProviderMatcher(model_name=MATCHING_CONFIG['model_name'])
    providers_df = load_providers(PRESTATAIRES_CSV)
    matcher.load_providers(providers_df)
    matcher.encode_providers()
    
    # 2. Charger les besoins (enrichis avec NER si disponible)
    from utils import load_needs
    import os
    besoins_file = BESOINS_ENRICHIS_CSV if MATCHING_CONFIG.get('use_enriched') and os.path.exists(BESOINS_ENRICHIS_CSV) else BESOINS_CSV
    needs_df = load_needs(besoins_file)
    print(f">> Fichier utilisé: {besoins_file.name}")
    
    # 3. Matcher le premier besoin
    print(f"\n>> Matching du premier besoin du CSV...")
    first_need = needs_df.iloc[0]
    
    # Afficher le besoin
    print(f"\nBesoin sélectionné:")
    print(f"  - Catégorie: {first_need['Categorie_Majeure']}")
    print(f"  - Type: {first_need['Sous_Categorie']}")
    print(f"  - Urgence: {first_need['Niveau_Urgence']}")
    print(f"  - Message: {first_need['Message_Utilisateur']}")
    
    # Trouver les matches
    results = matcher.match_need_row(first_need, top_k=3, threshold=0.25)
    
    # Afficher
    display_matches(results)
    
    return matcher, results


def example_batch_matching():
    """Exemple: Matcher tous les besoins du CSV en batch"""
    
    print("\n" + "="*80)
    print("EXEMPLE 3: BATCH MATCHING DE TOUS LES BESOINS")
    print("="*80)
    
    # 1. Initialiser
    matcher = ProviderMatcher(model_name=MATCHING_CONFIG['model_name'])
    providers_df = load_providers(PRESTATAIRES_CSV)
    matcher.load_providers(providers_df)
    matcher.encode_providers()
    
    # 2. Charger tous les besoins (enrichis avec NER si disponible)
    from utils import load_needs
    import os
    besoins_file = BESOINS_ENRICHIS_CSV if MATCHING_CONFIG.get('use_enriched') and os.path.exists(BESOINS_ENRICHIS_CSV) else BESOINS_CSV
    needs_df = load_needs(besoins_file)
    print(f">> Fichier utilisé: {besoins_file.name}")
    
    # 3. Batch matching (limiter à 5 pour l'exemple)
    sample_needs = needs_df.head(5)
    batch_results = matcher.batch_match(sample_needs, top_k=3, threshold=0.25)
    
    # 4. Afficher un résumé
    print(f"\n>> Résultats du batch matching:")
    for need_idx, matches in batch_results:
        need = needs_df.loc[need_idx]
        print(f"\nBesoin #{need_idx}: {need['Sous_Categorie']}")
        print(f"  - {len(matches)} prestataires trouvés")
        if len(matches) > 0:
            best_match = matches.iloc[0]
            print(f"  - Meilleur: {best_match['Nom_Entreprise']} ({best_match['similarity_score']*100:.1f}%)")
    
    return matcher, batch_results


def match_all_needs_to_csv():
    """Matcher TOUS les besoins du CSV et sauvegarder les résultats"""
    
    print("\n" + "="*80)
    print("MATCHING COMPLET DE TOUS LES BESOINS")
    print("="*80)
    
    # 1. Initialiser
    matcher = ProviderMatcher(model_name=MATCHING_CONFIG['model_name'])
    providers_df = load_providers(PRESTATAIRES_CSV)
    matcher.load_providers(providers_df)
    
    print("\n>> Encodage des prestataires...")
    matcher.encode_providers()
    
    # 2. Charger tous les besoins (enrichis avec NER si disponible)
    from utils import load_needs
    import os
    besoins_file = BESOINS_ENRICHIS_CSV if MATCHING_CONFIG.get('use_enriched') and os.path.exists(BESOINS_ENRICHIS_CSV) else BESOINS_CSV
    needs_df = load_needs(besoins_file)
    print(f">> Fichier utilisé: {besoins_file.name}")
    print(f">> {len(needs_df)} besoins chargés")
    
    # 3. Batch matching de TOUS les besoins
    print(f"\n>> Lancement du matching pour les {len(needs_df)} besoins...")
    batch_results = matcher.batch_match(needs_df, top_k=MATCHING_CONFIG['top_k'], threshold=MATCHING_CONFIG['threshold'])
    
    # 4. Créer un DataFrame consolidé avec tous les résultats
    import pandas as pd
    all_matches = []
    
    print("\n>> Consolidation des résultats...")
    for need_idx, matches in batch_results:
        need = needs_df.loc[need_idx]
        
        if len(matches) > 0:
            for rank, (_, match_row) in enumerate(matches.iterrows(), 1):
                match_record = {
                    'besoin_id': need_idx,
                    'besoin_categorie': need['Categorie_Majeure'],
                    'besoin_sous_categorie': need['Sous_Categorie'],
                    'besoin_message': need['Message_Utilisateur'],
                    'besoin_urgence': need['Niveau_Urgence'],
                    'rank': rank,
                    'prestataire_nom': match_row['Nom_Entreprise'],
                    'prestataire_domaines': match_row['Domaines_Expertise'],
                    'prestataire_disponibilite': match_row['Disponibilite'],
                    'score_similarite': round(match_row['similarity_score'] * 100, 2)
                }
                all_matches.append(match_record)
        else:
            # Besoin sans match
            match_record = {
                'besoin_id': need_idx,
                'besoin_categorie': need['Categorie_Majeure'],
                'besoin_sous_categorie': need['Sous_Categorie'],
                'besoin_message': need['Message_Utilisateur'],
                'besoin_urgence': need['Niveau_Urgence'],
                'rank': 0,
                'prestataire_nom': 'AUCUN MATCH',
                'prestataire_domaines': '',
                'prestataire_disponibilite': '',
                'score_similarite': 0.0
            }
            all_matches.append(match_record)
    
    results_df = pd.DataFrame(all_matches)
    
    # 5. Sauvegarder dans un CSV
    output_path = PROCESSED_DATA_DIR / 'matching_complet_tous_besoins.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
    
    # 6. Afficher le résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ DU MATCHING COMPLET")
    print(f"{'='*80}")
    print(f"\nTotal besoins traités: {len(needs_df)}")
    print(f"Total matches trouvés: {len(results_df[results_df['rank'] > 0])}")
    print(f"Besoins sans match: {len(results_df[results_df['rank'] == 0])}")
    print(f"\nScore moyen: {results_df[results_df['rank'] > 0]['score_similarite'].mean():.2f}%")
    print(f"Score minimum: {results_df[results_df['rank'] > 0]['score_similarite'].min():.2f}%")
    print(f"Score maximum: {results_df[results_df['rank'] > 0]['score_similarite'].max():.2f}%")
    
    print(f"\n>> Résultats sauvegardés dans: {output_path}")
    
    # 7. Afficher quelques exemples
    print(f"\n{'='*80}")
    print("EXEMPLES DE MATCHING (5 premiers besoins)")
    print(f"{'='*80}")
    
    for need_idx in needs_df.head(5).index:
        need = needs_df.loc[need_idx]
        need_matches = results_df[results_df['besoin_id'] == need_idx]
        
        print(f"\nBesoin #{need_idx}: {need['Sous_Categorie']}")
        print(f"Message: {need['Message_Utilisateur'][:80]}...")
        print(f"Matches trouvés: {len(need_matches[need_matches['rank'] > 0])}")
        
        if len(need_matches[need_matches['rank'] > 0]) > 0:
            best_match = need_matches.iloc[0]
            print(f"  -> Meilleur: {best_match['prestataire_nom']} ({best_match['score_similarite']}%)")
    
    return matcher, results_df


def interactive_mode():
    """Mode interactif pour tester des requêtes"""
    
    print("\n" + "="*80)
    print("MODE INTERACTIF - MATCHING DE PRESTATAIRES")
    print("="*80)
    
    # Initialiser le matcher
    matcher = ProviderMatcher(model_name=MATCHING_CONFIG['model_name'])
    providers_df = load_providers(PRESTATAIRES_CSV)
    matcher.load_providers(providers_df)
    matcher.encode_providers()
    
    print("\nLe système est prêt. Vous pouvez maintenant entrer vos requêtes.")
    print("Tapez 'quit' pour quitter.\n")
    
    while True:
        user_input = input("Votre besoin: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Au revoir !")
            break
        
        if not user_input.strip():
            continue
        
        # Faire le matching
        # Note: impact_geo=1 par défaut (services locaux)
        # Pour personnaliser, demander à l'utilisateur ou détecter automatiquement
        results = matcher.find_matches(user_input, top_k=3, threshold=0.2, impact_geo=1)
        
        # Afficher
        display_matches(results, max_display=3)
        print()


def main():
    """Fonction principale - choisir l'exemple à exécuter"""
    
    print("\n" + "="*80)
    print("SYSTEME DE MATCHING PRESTATAIRES - ZEUS ARMEE")
    print("="*80)
    
    print("\nQuel exemple voulez-vous exécuter ?")
    print("1. Matching pour une demande unique")
    print("2. Matching depuis le CSV besoins")
    print("3. Batch matching de tous les besoins (limité à 5)")
    print("4. Mode interactif")
    print("5. Matching COMPLET de tous les besoins + export CSV")
    print("0. Tout exécuter")
    
    choice = input("\nVotre choix (0-5): ").strip()
    
    if choice == '1':
        example_single_request()
    elif choice == '2':
        example_from_csv()
    elif choice == '3':
        example_batch_matching()
    elif choice == '4':
        interactive_mode()
    elif choice == '5':
        match_all_needs_to_csv()
    elif choice == '0':
        # Exécuter tous les exemples
        example_single_request()
        example_from_csv()
        example_batch_matching()
    else:
        print("Choix invalide. Exécution de l'exemple 1 par défaut.")
        example_single_request()


if __name__ == "__main__":
    main()
