"""
Script pour enrichir le fichier besoins_v2.csv avec les entités NER extraites
Génère un fichier besoins_v2_enrichis.csv avec colonnes supplémentaires
"""

import sys
from pathlib import Path
import pandas as pd

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, BESOINS_V2_CSV
from utils.ner_extractor import NERExtractor


def enrichir_besoins_v2():
    """
    Enrichit le fichier besoins_v2.csv avec les entités NER
    """
    print("\n" + "="*80)
    print("ENRICHISSEMENT DES BESOINS V2 AVEC NER")
    print("="*80)
    
    # 1. Charger les besoins v2
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
            if entities['metiers']:
                print(f"   -> Métiers: {', '.join(entities['metiers'])}")
            if entities['mots_cles']:
                print(f"   -> Mots-clés: {', '.join(entities['mots_cles'][:3])}")
        
        # Barre de progression
        if (idx + 1) % 10 == 0:
            print(f"   Progression: {idx + 1}/{len(besoins_df)} besoins traités")
    
    # 4. Créer un DataFrame avec toutes les entités
    print("\n>> Création du DataFrame enrichi...")
    entities_df = pd.DataFrame(entities_list)
    
    # 5. Fusionner avec le DataFrame original
    besoins_enrichis = besoins_df.copy()
    
    # Ajouter toutes les colonnes d'entités
    for col in entities_df.columns:
        if col != 'besoin_id':
            besoins_enrichis[col] = entities_df[col]
    
    # 6. Sauvegarder le fichier enrichi
    output_path = RAW_DATA_DIR / 'besoins_v2_enrichis.csv'
    besoins_enrichis.to_csv(output_path, index=False, sep=';', encoding='utf-8')
    
    # 7. Afficher le résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ DE L'ENRICHISSEMENT V2")
    print(f"{'='*80}")
    print(f"\nFichier d'entrée: {BESOINS_V2_CSV}")
    print(f"Fichier de sortie: {output_path}")
    print(f"Nombre de besoins: {len(besoins_enrichis)}")
    print(f"Colonnes originales: {len(besoins_df.columns)}")
    print(f"Colonnes enrichies: {len(besoins_enrichis.columns)}")
    print(f"Nouvelles colonnes ajoutées: {len(besoins_enrichis.columns) - len(besoins_df.columns)}")
    
    # Statistiques sur les entités détectées
    print(f"\n{'='*80}")
    print("STATISTIQUES DES ENTITÉS DÉTECTÉES")
    print(f"{'='*80}")
    
    print(f"\nVilles détectées: {besoins_enrichis['ville_detectee'].notna().sum()}/{len(besoins_enrichis)}")
    print(f"Départements détectés: {besoins_enrichis['departement'].notna().sum()}/{len(besoins_enrichis)}")
    
    # Compter les métiers
    metiers_count = besoins_enrichis['metiers'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
    print(f"Total métiers détectés: {metiers_count}")
    
    # Compter les mots-clés
    mots_cles_count = besoins_enrichis['mots_cles'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
    print(f"Total mots-clés détectés: {mots_cles_count}")
    
    print(f"\n>> Fichier enrichi sauvegardé avec succès!")
    print(f">> Vous pouvez maintenant lancer le matching avec l'option 6")
    
    return besoins_enrichis


if __name__ == "__main__":
    enrichir_besoins_v2()
