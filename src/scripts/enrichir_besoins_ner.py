"""
Script pour enrichir le fichier besoins.csv avec les entités NER extraites
Génère un fichier besoins_enrichis.csv avec colonnes supplémentaires
"""

import sys
from pathlib import Path
import pandas as pd

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

from config import BESOINS_V2_CSV, RAW_DATA_DIR
from utils.ner_extractor import NERExtractor


def enrichir_besoins():
    """
    Enrichit le fichier besoins.csv avec les entités NER
    """
    print("\n" + "="*80)
    print("ENRICHISSEMENT DES BESOINS AVEC NER")
    print("="*80)
    
    # 1. Charger les besoins
    print(f"\n>> Chargement de {BESOINS_V2_CSV}...")
    besoins_df = pd.read_csv(BESOINS_V2_CSV, sep=';', encoding='utf-8')
    print(f"   {len(besoins_df)} besoins chargés")
    
    # 2. Initialiser l'extracteur NER
    ner = NERExtractor()
    
    # 3. Extraire les entités pour chaque besoin
    print("\n>> Extraction des entités NER...")
    
    entities_list = []
    for idx, row in besoins_df.iterrows():
        message = row['Message_Utilisateur']
        niveau_urgence = row.get('Niveau_Urgence', None)
        
        # Extraction complète
        entities = ner.extract_all(message, niveau_urgence)
        entities['besoin_id'] = idx
        entities_list.append(entities)
        
        # Afficher quelques exemples
        if idx < 5:
            print(f"\n   Besoin #{idx}: {row['Sous_Categorie']}")
            print(f"   Message: {message[:80]}...")
            print(f"   -> Ville: {entities['ville_detectee']}")
            print(f"   -> Horizon: {entities['horizon_temporel']}")
            print(f"   -> Urgence: {entities['urgence_deduite']}")
            print(f"   -> Contraintes: {entities['contraintes_matching']}")
    
    # 4. Créer DataFrame avec entités
    entities_df = pd.DataFrame(entities_list)
    
    # 5. Fusionner avec besoins originaux
    besoins_enrichis = besoins_df.copy()
    besoins_enrichis['Ville_Detectee'] = entities_df['ville_detectee']
    besoins_enrichis['Date_Detectee'] = entities_df['date_detectee']
    besoins_enrichis['Horizon_Temporel_Normalise'] = entities_df['horizon_temporel']
    besoins_enrichis['Jours_Estimation'] = entities_df['jours_estimation']
    besoins_enrichis['Urgence_Deduite'] = entities_df['urgence_deduite']
    besoins_enrichis['Contrainte_Ville'] = entities_df['contraintes_matching'].apply(lambda x: x['ville'])
    besoins_enrichis['Contrainte_Disponibilite'] = entities_df['contraintes_matching'].apply(lambda x: x['disponibilite'])
    
    # 6. Sauvegarder le fichier enrichi
    output_path = RAW_DATA_DIR / 'besoins_enrichis.csv'
    besoins_enrichis.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
    
    print(f"\n{'='*80}")
    print("RÉSUMÉ DE L'ENRICHISSEMENT")
    print(f"{'='*80}")
    
    # Statistiques
    print(f"\nBesoins avec ville détectée: {besoins_enrichis['Ville_Detectee'].notna().sum()}/{len(besoins_enrichis)}")
    print(f"Besoins avec date détectée: {besoins_enrichis['Date_Detectee'].notna().sum()}/{len(besoins_enrichis)}")
    print(f"Besoins avec horizon normalisé: {besoins_enrichis['Horizon_Temporel_Normalise'].notna().sum()}/{len(besoins_enrichis)}")
    
    print("\nRépartition par urgence déduite:")
    print(besoins_enrichis['Urgence_Deduite'].value_counts())
    
    print("\nRépartition par horizon temporel:")
    print(besoins_enrichis['Horizon_Temporel_Normalise'].value_counts())
    
    print("\nTop 5 villes détectées:")
    print(besoins_enrichis['Ville_Detectee'].value_counts().head())
    
    print(f"\n>> Fichier enrichi sauvegardé: {output_path}")
    
    # 7. Sauvegarder aussi en JSON pour inspection détaillée
    output_json = RAW_DATA_DIR / 'besoins_enrichis.json'
    
    # Créer structure JSON avec toutes les infos
    json_records = []
    for idx, row in besoins_enrichis.iterrows():
        record = {
            'besoin_id': int(idx),
            'categorie_majeure': row['Categorie_Majeure'],
            'sous_categorie': row['Sous_Categorie'],
            'message_utilisateur': row['Message_Utilisateur'],
            'niveau_urgence_original': row['Niveau_Urgence'],
            'entites_ner': {
                'ville_detectee': row['Ville_Detectee'] if pd.notna(row['Ville_Detectee']) else None,
                'date_detectee': row['Date_Detectee'] if pd.notna(row['Date_Detectee']) else None,
                'horizon_temporel': row['Horizon_Temporel_Normalise'] if pd.notna(row['Horizon_Temporel_Normalise']) else None,
                'jours_estimation': int(row['Jours_Estimation']) if pd.notna(row['Jours_Estimation']) else None,
                'urgence_deduite': row['Urgence_Deduite']
            },
            'contraintes_matching': {
                'ville': row['Contrainte_Ville'],
                'disponibilite': row['Contrainte_Disponibilite']
            }
        }
        json_records.append(record)
    
    import json
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)
    
    print(f">> Fichier JSON sauvegardé: {output_json}")
    
    return besoins_enrichis


def afficher_exemples_matching(besoins_enrichis: pd.DataFrame):
    """
    Affiche des exemples de règles de matching basées sur les entités NER
    """
    print("\n" + "="*80)
    print("EXEMPLES DE RÈGLES DE MATCHING")
    print("="*80)
    
    ner = NERExtractor()
    
    # Exemple 1: Besoin immédiat avec ville
    exemple_1 = besoins_enrichis[
        (besoins_enrichis['Urgence_Deduite'] == 'IMMEDIATE') &
        (besoins_enrichis['Ville_Detectee'].notna())
    ].iloc[0]
    
    print("\n📌 Exemple 1 - Besoin immédiat avec ville")
    print(f"   Besoin: {exemple_1['Sous_Categorie']}")
    print(f"   Message: {exemple_1['Message_Utilisateur'][:100]}...")
    print(f"\n   Entités extraites:")
    print(f"   - Ville: {exemple_1['Ville_Detectee']}")
    print(f"   - Urgence: {exemple_1['Urgence_Deduite']}")
    print(f"   - Contrainte dispo: {exemple_1['Contrainte_Disponibilite']}")
    print(f"\n   Règles de matching applicables:")
    print(f"   ✓ Filtrer prestataires avec disponibilité: 24/7")
    print(f"   ✓ Privilégier prestataires de {exemple_1['Ville_Detectee']} (boost +30%)")
    print(f"   ✓ Accepter prestataires nationaux/en ligne")
    
    # Exemple 2: Besoin planifié
    exemple_2 = besoins_enrichis[
        besoins_enrichis['Urgence_Deduite'] == 'PLANNED'
    ].iloc[0]
    
    print("\n📌 Exemple 2 - Besoin planifié")
    print(f"   Besoin: {exemple_2['Sous_Categorie']}")
    print(f"   Message: {exemple_2['Message_Utilisateur'][:100]}...")
    print(f"\n   Entités extraites:")
    print(f"   - Date: {exemple_2['Date_Detectee']}")
    print(f"   - Horizon: {exemple_2['Horizon_Temporel_Normalise']}")
    print(f"   - Jours: {exemple_2['Jours_Estimation']}")
    print(f"\n   Règles de matching applicables:")
    print(f"   ✓ Tous les prestataires éligibles (pas de contrainte urgence)")
    print(f"   ✓ Privilégier qualité et avis clients")
    print(f"   ✓ Match géographique flexible")


if __name__ == "__main__":
    # Enrichir les besoins
    besoins_enrichis = enrichir_besoins()
    
    # Afficher exemples de matching
    afficher_exemples_matching(besoins_enrichis)
    
    print("\n" + "="*80)
    print("TERMINÉ - Les fichiers enrichis sont prêts pour le matching !")
    print("="*80)
