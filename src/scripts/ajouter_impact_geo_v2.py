"""
Script pour ajouter la colonne 'Impact_Geo' au dataset besoins_v2_enrichis.csv

Impact géographique:
- 0: Aucun impact (services en ligne, finance, assurance)
- 1: Impact modéré (logement, emploi local, démarches physiques)
- 2: Impact très fort (urgences, garde d'enfants, dépannages)
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR

# Réutiliser le même mapping que pour besoins.csv
from ajouter_impact_geo import IMPACT_GEO_MAPPING


def ajouter_impact_geo_v2():
    """
    Ajoute la colonne Impact_Geo au dataset besoins_v2_enrichis.csv
    Utilise le même mapping basé sur (Categorie_Majeure, Sous_Categorie)
    """
    
    # Charger le fichier enrichi v2
    besoins_file = RAW_DATA_DIR / "besoins_v2_enrichis.csv"
    
    if not besoins_file.exists():
        print(f"❌ Fichier non trouvé: {besoins_file}")
        print("Veuillez d'abord exécuter enrichir_besoins_v2_ner.py")
        return
    
    print(f"📖 Chargement de {besoins_file}...")
    df = pd.read_csv(besoins_file, sep=';', encoding='utf-8')
    
    print(f"   {len(df)} besoins chargés")
    
    # Vérifier si la colonne existe déjà
    if 'Impact_Geo' in df.columns:
        print("⚠️  La colonne 'Impact_Geo' existe déjà. Suppression pour réinitialisation...")
        df = df.drop(columns=['Impact_Geo'])
    
    # Créer une colonne tuple pour le mapping
    print("\n🗺️  Application du mapping (Categorie_Majeure, Sous_Categorie) → Impact_Geo...")
    df['_mapping_key'] = list(zip(df['Categorie_Majeure'], df['Sous_Categorie']))
    
    # Appliquer le mapping
    df['Impact_Geo'] = df['_mapping_key'].map(IMPACT_GEO_MAPPING)
    
    # Supprimer la colonne temporaire
    df = df.drop(columns=['_mapping_key'])
    
    # Vérifier les valeurs manquantes
    missing = df[df['Impact_Geo'].isna()]
    if not missing.empty:
        print(f"\n⚠️  {len(missing)} combinaisons sans mapping:")
        print(f"\n   Combinaisons manquantes à ajouter dans IMPACT_GEO_MAPPING:")
        for _, row in missing[['Categorie_Majeure', 'Sous_Categorie']].drop_duplicates().iterrows():
            print(f"   ('{row['Categorie_Majeure']}', '{row['Sous_Categorie']}'): 1,  # À définir: 0, 1 ou 2")
        
        # Attribuer une valeur par défaut de 1 (modéré) pour les non mappés
        print(f"\n⚡ Valeur par défaut: Impact_Geo = 1 (modéré) pour les {len(missing)} combinaisons non mappées")
        df.loc[df['Impact_Geo'].isna(), 'Impact_Geo'] = 1
    
    # Statistiques
    print("\n📊 Répartition des impacts géographiques:")
    impact_counts = df['Impact_Geo'].value_counts().sort_index()
    for impact, count in impact_counts.items():
        pct = (count / len(df)) * 100
        label = {0: "NUL (services en ligne)", 1: "MODÉRÉ (local)", 2: "TRÈS FORT (critique)"}[impact]
        print(f"   Impact {impact} ({label}): {count} besoins ({pct:.1f}%)")
    
    # Sauvegarder
    output_file = RAW_DATA_DIR / "besoins_v2_enrichis.csv"
    print(f"\n💾 Sauvegarde dans {output_file}...")
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    
    print(f"\n✅ Colonne 'Impact_Geo' ajoutée avec succès!")
    print(f"   Total: {len(df)} besoins enrichis")
    
    # Afficher quelques exemples par catégorie
    print("\n📋 Exemples de résultats:")
    examples = df[['Categorie_Majeure', 'Sous_Categorie', 'Impact_Geo']].drop_duplicates().sort_values(['Impact_Geo', 'Categorie_Majeure'])
    for _, row in examples.head(20).iterrows():
        cat = row['Categorie_Majeure'][:25].ljust(25)
        sous_cat = row['Sous_Categorie'][:30].ljust(30)
        print(f"   {cat} | {sous_cat} → Impact_Geo = {int(row['Impact_Geo'])}")


if __name__ == "__main__":
    ajouter_impact_geo_v2()
