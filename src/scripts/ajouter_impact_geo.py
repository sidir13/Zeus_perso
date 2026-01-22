"""
Script pour ajouter la colonne 'Impact_Geo' au dataset besoins_enrichis.csv

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

# Mapping métier basé sur (Categorie_Majeure, Sous_Categorie)
# Clé = (Categorie_Majeure, Sous_Categorie), Valeur = impact_geo (0, 1, ou 2)
IMPACT_GEO_MAPPING = {
    # ==================== IMPACT 2 : Proximité CRITIQUE ====================
    # Besoins de dernière minute
    ('Besoins de dernière minute', 'Garde d\'enfant'): 2,
    ('Besoins de dernière minute', 'Électroménager'): 2,
    ('Besoins de dernière minute', 'Transport express'): 2,
    ('Besoins de dernière minute', 'Plomberie urgente'): 2,
    ('Besoins de dernière minute', 'Coiffure'): 2,
    ('Besoins de dernière minute', 'Pressing express'): 2,
    ('Besoins de dernière minute', 'Serrurier'): 2,
    ('Besoins de dernière minute', 'Vétérinaire'): 2,
    ('Besoins de dernière minute', 'Informatique'): 2,
    ('Besoins de dernière minute', 'Traiteur'): 2,
    
    # Famille - urgences
    ('Famille', 'Crèche ou nounou'): 2,
    
    # Santé - urgences
    ('Santé', 'Dentiste d\'urgence'): 2,
    ('Santé', 'Kiné urgence'): 2,
    
    # Véhicule - urgences
    ('Véhicule', 'Réparation urgente'): 2,
    ('Véhicule', 'Dépannage remorquage'): 2,
    ('Véhicule', 'Location courte durée'): 2,
    
    # Travaux - urgences
    ('Travaux', 'Plomberie urgente'): 2,
    ('Travaux', 'Chauffage'): 2,
    
    # Logement - urgences
    ('Logement et Installation', 'Déménagement'): 2,
    ('Logement et Installation', 'État des lieux'): 2,
    ('Logement et Installation', 'Emménagement assisté'): 2,
    
    # ==================== IMPACT 1 : Proximité UTILE ====================
    # Logement et Installation
    ('Logement et Installation', 'Location meublée'): 1,
    ('Logement et Installation', 'Recherche colocation'): 1,
    ('Logement et Installation', 'Recherche logement social'): 1,
    ('Logement et Installation', 'Stockage temporaire'): 1,
    ('Logement et Installation', 'Garde-meuble'): 1,
    ('Logement et Installation', 'Aménagement'): 1,
    ('Logement et Installation', 'Diagnostic immobilier'): 1,
    ('Logement et Installation', 'Bail meublé'): 1,
    ('Logement et Installation', 'Construction maison retraite'): 1,
    
    # Famille
    ('Famille', 'Scolarité'): 1,
    ('Famille', 'Activités périscolaires'): 1,
    ('Famille', 'Aide aux devoirs'): 1,
    ('Famille', 'Garde animaux'): 1,
    ('Famille', 'Mode de garde partagé'): 1,
    ('Famille', 'Activités enfants'): 1,
    ('Famille', 'Soutien scolaire'): 1,
    ('Famille', 'Périscolaire'): 1,
    
    # Administratif
    ('Administratif', 'Carte grise'): 1,
    ('Administratif', 'Passeport express'): 1,
    ('Administratif', 'Titre de séjour conjoint'): 1,
    ('Administratif', 'Changement situation familiale'): 1,
    ('Administratif', 'Déclaration impôts'): 1,
    ('Administratif', 'Carte Vitale'): 1,
    ('Administratif', 'Naturalisation conjoint'): 1,
    ('Administratif', 'Permis de conduire'): 1,
    
    # Véhicule
    ('Véhicule', 'Contrôle technique'): 1,
    ('Véhicule', 'Achat véhicule'): 1,
    ('Véhicule', 'Reprogrammation moteur'): 1,
    ('Véhicule', 'Expertise accident'): 1,
    ('Véhicule', 'Révision technique'): 1,
    ('Véhicule', 'Pneumatiques'): 1,
    ('Véhicule', 'Assurance auto'): 0,
    
    # Travaux
    ('Travaux', 'Installation fibre'): 1,
    ('Travaux', 'Rénovation avant vente'): 1,
    ('Travaux', 'Mise en conformité logement'): 1,
    ('Travaux', 'Peinture intérieure'): 1,
    ('Travaux', 'Toiture'): 1,
    
    # Emploi
    ('Emploi', 'Recherche emploi conjoint'): 1,
    ('Emploi', 'Bilan de compétences'): 1,
    ('Emploi', 'Formation conjoint'): 1,
    ('Emploi', 'CV et lettre motivation'): 1,
    
    # Santé - non urgent
    ('Santé', 'Ophtalmologue'): 1,
    ('Santé', 'Médecin généraliste'): 1,
    ('Santé', 'Podologue'): 1,
    ('Santé', 'Orthodontie'): 1,
    ('Santé', 'Allergologue'): 1,
    
    # Retraite
    ('Retraite', 'Construction maison retraite'): 1,
    ('Retraite', 'Maison de retraite'): 1,
    
    # Formation - présentiel
    ('Formation', 'Permis poids lourd'): 1,
    ('Formation', 'Habilitation électrique'): 1,
    ('Formation', 'Sécurité incendie'): 1,
    
    # Épargne - parfois physique
    ('Épargne', 'Placement financier'): 1,
    ('Épargne', 'Assurance-vie'): 1,
    
    # ==================== IMPACT 0 : Services en ligne ====================
    # Banque et financement
    ('Banque et financement', 'Prêt immobilier'): 0,
    ('Banque et financement', 'Prêt travaux'): 0,
    ('Banque et financement', 'Regroupement crédits'): 0,
    ('Banque et financement', 'Placement financier'): 0,
    ('Banque et financement', 'Crédit consommation'): 0,
    ('Banque et financement', 'Découvert bancaire'): 0,
    ('Banque et financement', 'Épargne salariale'): 0,
    ('Banque et financement', 'Rachat assurance prêt'): 0,
    
    # Assurance
    ('Assurance', 'Mutuelle santé'): 0,
    ('Assurance', 'Assurance habitation'): 0,
    ('Assurance', 'Assurance auto jeune conducteur'): 0,
    ('Assurance', 'Prévoyance'): 0,
    ('Assurance', 'Responsabilité civile'): 0,
    ('Assurance', 'Assurance emprunteur'): 0,
    ('Assurance', 'Assurance auto'): 0,
    ('Assurance', 'Garantie accident vie'): 0,
    
    # Formation - en ligne
    ('Formation', 'Reconversion professionnelle'): 0,
    ('Formation', 'Certification professionnelle'): 0,
    ('Formation', 'Langue étrangère'): 0,
    
    # Soutien psychologique
    ('Soutien psychologique', 'Accompagnement familial'): 0,
    ('Soutien psychologique', 'Gestion stress opérationnel'): 0,
    ('Soutien psychologique', 'Thérapie de couple'): 0,
    ('Soutien psychologique', 'Traumatisme SSPT'): 0,
    
    # Retraite - planification
    ('Retraite', 'Préparation retraite'): 0,
    ('Retraite', 'Calcul droits'): 0,
    
    # Emploi - en ligne
    ('Emploi', 'Aide à la création entreprise'): 0,
    ('Emploi', 'Reconversion militaire'): 0,
}


def ajouter_impact_geo():
    """
    Ajoute la colonne Impact_Geo au dataset besoins_enrichis.csv
    Utilise un mapping basé sur (Categorie_Majeure, Sous_Categorie)
    """
    
    # Charger le fichier enrichi
    besoins_file = RAW_DATA_DIR / "besoins_enrichis.csv"
    
    if not besoins_file.exists():
        print(f"❌ Fichier non trouvé: {besoins_file}")
        print("Veuillez d'abord exécuter enrichir_besoins_ner.py")
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
        for _, row in missing[['Categorie_Majeure', 'Sous_Categorie']].drop_duplicates().iterrows():
            print(f"   - ({row['Categorie_Majeure']}, {row['Sous_Categorie']})")
        print("\n❌ Veuillez compléter le mapping dans IMPACT_GEO_MAPPING")
        print("   Format: ('Categorie_Majeure', 'Sous_Categorie'): impact_geo")
        return
    
    # Statistiques
    print("\n📊 Répartition des impacts géographiques:")
    impact_counts = df['Impact_Geo'].value_counts().sort_index()
    for impact, count in impact_counts.items():
        pct = (count / len(df)) * 100
        label = {0: "NUL (services en ligne)", 1: "MODÉRÉ (local)", 2: "TRÈS FORT (critique)"}[impact]
        print(f"   Impact {impact} ({label}): {count} besoins ({pct:.1f}%)")
    
    # Sauvegarder
    output_file = RAW_DATA_DIR / "besoins_enrichis.csv"
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
    ajouter_impact_geo()
