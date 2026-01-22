"""
Analyse des données du fichier besoins.csv
Génère des statistiques et visualisations pour comprendre les besoins des militaires
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import BESOINS_CSV, PROCESSED_DATA_DIR

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_data():
    """Charge le fichier CSV des besoins"""
    print(f"Chargement des données depuis : {BESOINS_CSV}")
    df = pd.read_csv(BESOINS_CSV, sep=';', encoding='utf-8')
    print(f">> {len(df)} lignes chargées\n")
    return df

def display_basic_info(df):
    """Affiche les informations de base sur le dataset"""
    print("="*80)
    print("INFORMATIONS GÉNÉRALES")
    print("="*80)
    print(f"\nNombre total de besoins : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")
    print(f"\nColonnes : {list(df.columns)}")
    print(f"\nTypes de données :")
    print(df.dtypes)
    print(f"\n\nValeurs manquantes :")
    print(df.isnull().sum())
    print(f"\n\nPremières lignes :")
    print(df.head(3))

def analyze_categories(df):
    """Analyse des catégories majeures"""
    print("\n" + "="*80)
    print("ANALYSE DES CATÉGORIES MAJEURES")
    print("="*80)
    
    cat_counts = df['Categorie_Majeure'].value_counts()
    print(f"\nRépartition des {len(cat_counts)} catégories majeures :")
    print(cat_counts)
    print(f"\nPourcentages :")
    print((cat_counts / len(df) * 100).round(2))
    
    # Visualisation
    plt.figure(figsize=(14, 8))
    cat_counts.plot(kind='barh', color='steelblue')
    plt.title('Répartition des besoins par catégorie majeure', fontsize=16, fontweight='bold')
    plt.xlabel('Nombre de besoins')
    plt.ylabel('Catégorie majeure')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'categories_majeures.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : categories_majeures.png")
    plt.show()
    plt.close()

def analyze_subcategories(df):
    """Analyse des sous-catégories"""
    print("\n" + "="*80)
    print("ANALYSE DES SOUS-CATÉGORIES")
    print("="*80)
    
    subcat_counts = df['Sous_Categorie'].value_counts()
    print(f"\nNombre de sous-catégories différentes : {len(subcat_counts)}")
    print(f"\nTop 10 des sous-catégories les plus fréquentes :")
    print(subcat_counts.head(10))
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    subcat_counts.head(15).plot(kind='barh', color='coral')
    plt.title('Top 15 des sous-catégories de besoins', fontsize=16, fontweight='bold')
    plt.xlabel('Nombre de besoins')
    plt.ylabel('Sous-catégorie')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'sous_categories.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : sous_categories.png")
    plt.show()
    plt.close()

def analyze_urgency(df):
    """Analyse des niveaux d'urgence"""
    print("\n" + "="*80)
    print("ANALYSE DES NIVEAUX D'URGENCE")
    print("="*80)
    
    urgency_counts = df['Niveau_Urgence'].value_counts()
    print(f"\nRépartition des niveaux d'urgence :")
    print(urgency_counts)
    print(f"\nPourcentages :")
    print((urgency_counts / len(df) * 100).round(2))
    
    # Visualisation camembert
    plt.figure(figsize=(10, 8))
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
    urgency_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Répartition des besoins par niveau d\'urgence', fontsize=16, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'niveaux_urgence.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : niveaux_urgence.png")
    plt.show()
    plt.close()

def analyze_urgency_by_category(df):
    """Analyse croisée urgence × catégorie"""
    print("\n" + "="*80)
    print("ANALYSE CROISÉE : URGENCE × CATÉGORIE")
    print("="*80)
    
    cross_tab = pd.crosstab(df['Categorie_Majeure'], df['Niveau_Urgence'])
    print(f"\nTableau croisé :")
    print(cross_tab)
    
    # Visualisation
    plt.figure(figsize=(14, 8))
    cross_tab.plot(kind='bar', stacked=True)
    plt.title('Distribution des niveaux d\'urgence par catégorie', fontsize=16, fontweight='bold')
    plt.xlabel('Catégorie majeure')
    plt.ylabel('Nombre de besoins')
    plt.legend(title='Niveau d\'urgence', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'urgence_par_categorie.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : urgence_par_categorie.png")
    plt.show()
    plt.close()

def analyze_messages(df):
    """Analyse des messages utilisateurs"""
    print("\n" + "="*80)
    print("ANALYSE DES MESSAGES UTILISATEURS")
    print("="*80)
    
    # Longueur des messages
    df['message_length'] = df['Message_Utilisateur'].str.len()
    
    print(f"\nStatistiques sur la longueur des messages :")
    print(f"  - Moyenne : {df['message_length'].mean():.0f} caractères")
    print(f"  - Médiane : {df['message_length'].median():.0f} caractères")
    print(f"  - Min : {df['message_length'].min()} caractères")
    print(f"  - Max : {df['message_length'].max()} caractères")
    
    # Mots-clés les plus fréquents
    all_words = ' '.join(df['Message_Utilisateur'].str.lower()).split()
    # Filtrer les mots courants
    stop_words = ['je', 'de', 'la', 'le', 'les', 'un', 'une', 'des', 'pour', 'et', 'en', 'à', 
                  'dans', 'sur', 'mon', 'ma', 'mes', 'du', 'au', 'avec', 'par', 'être', 'avoir']
    filtered_words = [word for word in all_words if len(word) > 3 and word not in stop_words]
    
    from collections import Counter
    word_freq = Counter(filtered_words).most_common(20)
    
    print(f"\n\nTop 20 des mots les plus fréquents dans les messages :")
    for word, count in word_freq:
        print(f"  - {word:20} : {count} fois")
    
    # Visualisation distribution longueur
    plt.figure(figsize=(12, 6))
    plt.hist(df['message_length'], bins=30, color='mediumpurple', edgecolor='black')
    plt.title('Distribution de la longueur des messages', fontsize=16, fontweight='bold')
    plt.xlabel('Nombre de caractères')
    plt.ylabel('Fréquence')
    plt.axvline(df['message_length'].mean(), color='red', linestyle='--', label=f'Moyenne: {df["message_length"].mean():.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'longueur_messages.png', dpi=300, bbox_inches='tight')
    print(f"\n>> Graphique sauvegardé : longueur_messages.png")
    plt.show()
    plt.close()

def generate_summary_report(df):
    """Génère un rapport récapitulatif"""
    print("\n" + "="*80)
    print("RAPPORT RÉCAPITULATIF")
    print("="*80)
    
    report = f"""
SYNTHESE DE L'ANALYSE DES BESOINS
{'='*80}

CHIFFRES CLES
  • Total de besoins analysés : {len(df)}
  • Nombre de catégories majeures : {df['Categorie_Majeure'].nunique()}
  • Nombre de sous-catégories : {df['Sous_Categorie'].nunique()}
  • Niveaux d'urgence : {df['Niveau_Urgence'].nunique()}

CATEGORIES LES PLUS DEMANDEES
{df['Categorie_Majeure'].value_counts().head(5).to_string()}

DISTRIBUTION D'URGENCE
{df['Niveau_Urgence'].value_counts().to_string()}
  
BESOINS URGENTS (Immédiat)
  • Nombre : {len(df[df['Niveau_Urgence'] == 'Immédiat'])}
  • Pourcentage : {len(df[df['Niveau_Urgence'] == 'Immédiat']) / len(df) * 100:.1f}%
  • Principales catégories :
{df[df['Niveau_Urgence'] == 'Immédiat']['Categorie_Majeure'].value_counts().head(5).to_string()}

CARACTERISTIQUES DES MESSAGES
  • Longueur moyenne : {df['message_length'].mean():.0f} caractères
  • Message le plus court : {df['message_length'].min()} caractères
  • Message le plus long : {df['message_length'].max()} caractères

INSIGHTS CLES
  1. Les besoins de dernière minute représentent {len(df[df['Categorie_Majeure'] == 'Besoins de dernière minute']) / len(df) * 100:.1f}% des demandes
  2. Les besoins \"Immédiat\" nécessitent une réactivité maximale des prestataires
  3. Forte demande en logement, emploi et famille liée à la mobilité militaire
  
{'='*80}
Fichiers générés dans : {PROCESSED_DATA_DIR}
  >> categories_majeures.png
  >> sous_categories.png
  >> niveaux_urgence.png
  >> urgence_par_categorie.png
  >> longueur_messages.png
  >> rapport_besoins.txt
{'='*80}
"""
    
    print(report)
    
    # Sauvegarder le rapport
    with open(PROCESSED_DATA_DIR / 'rapport_besoins.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n>> Rapport sauvegardé : rapport_besoins.txt")

def main():
    """Fonction principale d'exécution de l'analyse"""
    print("\n" + "="*80)
    print("ANALYSE DES DONNEES - BESOINS MILITAIRES")
    print("="*80 + "\n")
    
    # Créer le dossier processed s'il n'existe pas
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Charger les données
    df = load_data()
    
    # Exécuter les analyses
    display_basic_info(df)
    analyze_categories(df)
    analyze_subcategories(df)
    analyze_urgency(df)
    analyze_urgency_by_category(df)
    analyze_messages(df)
    generate_summary_report(df)
    
    print("\n" + "="*80)
    print("ANALYSE TERMINEE AVEC SUCCES !")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
