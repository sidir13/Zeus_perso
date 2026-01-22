"""
Analyse des données du fichier prestataires.csv
Génère des statistiques et visualisations pour comprendre l'offre de services
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import PRESTATAIRES_CSV, PROCESSED_DATA_DIR

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_data():
    """Charge le fichier CSV des prestataires"""
    print(f"Chargement des données depuis : {PRESTATAIRES_CSV}")
    df = pd.read_csv(PRESTATAIRES_CSV, sep=';', encoding='utf-8')
    print(f">> {len(df)} lignes chargées\n")
    return df

def display_basic_info(df):
    """Affiche les informations de base sur le dataset"""
    print("="*80)
    print("INFORMATIONS GENERALES")
    print("="*80)
    print(f"\nNombre total de prestataires : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")
    print(f"\nColonnes : {list(df.columns)}")
    print(f"\nTypes de données :")
    print(df.dtypes)
    print(f"\n\nValeurs manquantes :")
    print(df.isnull().sum())
    print(f"\n\nPremières lignes :")
    print(df.head(3))

def analyze_expertise_domains(df):
    """Analyse des domaines d'expertise"""
    print("\n" + "="*80)
    print("ANALYSE DES DOMAINES D'EXPERTISE")
    print("="*80)
    
    # Séparer les domaines multiples
    all_domains = []
    for domains in df['Domaines_Expertise']:
        all_domains.extend([d.strip() for d in domains.split(',')])
    
    domain_counts = pd.Series(all_domains).value_counts()
    print(f"\nNombre de domaines différents : {len(domain_counts)}")
    print(f"\nTop 15 des domaines d'expertise les plus fréquents :")
    print(domain_counts.head(15))
    
    # Visualisation
    plt.figure(figsize=(14, 8))
    domain_counts.head(20).plot(kind='barh', color='steelblue')
    plt.title('Top 20 des domaines d\'expertise des prestataires', fontsize=16, fontweight='bold')
    plt.xlabel('Nombre de prestataires')
    plt.ylabel('Domaine d\'expertise')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'domaines_expertise.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : domaines_expertise.png")
    plt.show()
    plt.close()
    
    return all_domains, domain_counts

def analyze_availability(df):
    """Analyse de la disponibilité"""
    print("\n" + "="*80)
    print("ANALYSE DE LA DISPONIBILITE")
    print("="*80)
    
    availability_counts = df['Disponibilite'].value_counts()
    print(f"\nRépartition des disponibilités :")
    print(availability_counts)
    print(f"\nPourcentages :")
    print((availability_counts / len(df) * 100).round(2))
    
    # Visualisation camembert
    plt.figure(figsize=(10, 8))
    colors = ['#ff6b6b', '#48dbfb', '#1dd1a1', '#feca57']
    availability_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Répartition des prestataires par disponibilité', fontsize=16, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'disponibilite_prestataires.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : disponibilite_prestataires.png")
    plt.show()
    plt.close()

def analyze_expertise_count(df):
    """Analyse du nombre de domaines par prestataire"""
    print("\n" + "="*80)
    print("ANALYSE DU NOMBRE DE DOMAINES PAR PRESTATAIRE")
    print("="*80)
    
    df['nb_domaines'] = df['Domaines_Expertise'].apply(lambda x: len(x.split(',')))
    
    print(f"\nStatistiques sur le nombre de domaines :")
    print(f"  - Moyenne : {df['nb_domaines'].mean():.2f} domaines")
    print(f"  - Médiane : {df['nb_domaines'].median():.0f} domaines")
    print(f"  - Min : {df['nb_domaines'].min()} domaine(s)")
    print(f"  - Max : {df['nb_domaines'].max()} domaines")
    
    domain_count_distribution = df['nb_domaines'].value_counts().sort_index()
    print(f"\nRépartition :")
    print(domain_count_distribution)
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    domain_count_distribution.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Distribution du nombre de domaines d\'expertise par prestataire', fontsize=16, fontweight='bold')
    plt.xlabel('Nombre de domaines')
    plt.ylabel('Nombre de prestataires')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'nb_domaines_par_prestataire.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : nb_domaines_par_prestataire.png")
    plt.show()
    plt.close()

def analyze_descriptions(df):
    """Analyse des descriptions de services"""
    print("\n" + "="*80)
    print("ANALYSE DES DESCRIPTIONS DE SERVICES")
    print("="*80)
    
    # Longueur des descriptions
    df['description_length'] = df['Description_Service'].str.len()
    
    print(f"\nStatistiques sur la longueur des descriptions :")
    print(f"  - Moyenne : {df['description_length'].mean():.0f} caractères")
    print(f"  - Médiane : {df['description_length'].median():.0f} caractères")
    print(f"  - Min : {df['description_length'].min()} caractères")
    print(f"  - Max : {df['description_length'].max()} caractères")
    
    # Mots-clés les plus fréquents
    all_words = ' '.join(df['Description_Service'].str.lower()).split()
    # Filtrer les mots courants
    stop_words = ['de', 'la', 'le', 'les', 'un', 'une', 'des', 'pour', 'et', 'en', 'à', 
                  'dans', 'sur', 'avec', 'par', 'être', 'avoir', 'du', 'au', 'aux']
    filtered_words = [word for word in all_words if len(word) > 3 and word not in stop_words]
    
    from collections import Counter
    word_freq = Counter(filtered_words).most_common(25)
    
    print(f"\n\nTop 25 des mots les plus fréquents dans les descriptions :")
    for word, count in word_freq:
        print(f"  - {word:20} : {count} fois")
    
    # Visualisation distribution longueur
    plt.figure(figsize=(12, 6))
    plt.hist(df['description_length'], bins=30, color='mediumpurple', edgecolor='black')
    plt.title('Distribution de la longueur des descriptions', fontsize=16, fontweight='bold')
    plt.xlabel('Nombre de caractères')
    plt.ylabel('Fréquence')
    plt.axvline(df['description_length'].mean(), color='red', linestyle='--', label=f'Moyenne: {df["description_length"].mean():.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'longueur_descriptions.png', dpi=300, bbox_inches='tight')
    print(f"\n>> Graphique sauvegardé : longueur_descriptions.png")
    plt.show()
    plt.close()

def analyze_specialized_services(df):
    """Analyse des services spécialisés (agrément, urgence, etc.)"""
    print("\n" + "="*80)
    print("ANALYSE DES SERVICES SPECIALISES")
    print("="*80)
    
    # Recherche de mots-clés dans les descriptions
    keywords = {
        'Agrément Défense': df['Description_Service'].str.contains('Défense|défense', case=False, na=False).sum(),
        'Service Urgence': df['Description_Service'].str.contains('urgence|urgent|24h|24/7', case=False, na=False).sum(),
        'Militaire': df['Description_Service'].str.contains('militaire|militaires', case=False, na=False).sum(),
        'Mutation': df['Description_Service'].str.contains('mutation', case=False, na=False).sum(),
        'Fonctionnaire': df['Description_Service'].str.contains('fonctionnaire', case=False, na=False).sum(),
        'Spécialisé': df['Description_Service'].str.contains('spécialisé|spécialiste', case=False, na=False).sum(),
        'Réactivité': df['Description_Service'].str.contains('réactiv|rapide', case=False, na=False).sum(),
    }
    
    keywords_df = pd.Series(keywords).sort_values(ascending=False)
    print(f"\nPrésence de mots-clés spécialisés dans les descriptions :")
    print(keywords_df)
    print(f"\nPourcentages :")
    print((keywords_df / len(df) * 100).round(1))
    
    # Visualisation
    plt.figure(figsize=(12, 7))
    keywords_df.plot(kind='barh', color='teal')
    plt.title('Présence de mots-clés spécialisés dans les descriptions', fontsize=16, fontweight='bold')
    plt.xlabel('Nombre de prestataires')
    plt.ylabel('Mots-clés')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'mots_cles_specialises.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : mots_cles_specialises.png")
    plt.show()
    plt.close()

def analyze_availability_by_domain(df, all_domains):
    """Analyse croisée disponibilité × domaines populaires"""
    print("\n" + "="*80)
    print("ANALYSE CROISEE : DISPONIBILITE × DOMAINES POPULAIRES")
    print("="*80)
    
    # Identifier les domaines les plus fréquents
    domain_counts = pd.Series(all_domains).value_counts()
    top_domains = domain_counts.head(10).index.tolist()
    
    # Créer une matrice pour chaque domaine
    availability_by_domain = {}
    for domain in top_domains:
        mask = df['Domaines_Expertise'].str.contains(domain, case=False, na=False)
        availability_by_domain[domain] = df[mask]['Disponibilite'].value_counts()
    
    cross_df = pd.DataFrame(availability_by_domain).T.fillna(0)
    print(f"\nTableau croisé (Top 10 domaines) :")
    print(cross_df)
    
    # Visualisation
    plt.figure(figsize=(14, 8))
    cross_df.plot(kind='bar', stacked=True)
    plt.title('Distribution de la disponibilité par domaine populaire', fontsize=16, fontweight='bold')
    plt.xlabel('Domaine d\'expertise')
    plt.ylabel('Nombre de prestataires')
    plt.legend(title='Disponibilité', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'disponibilite_par_domaine.png', dpi=300, bbox_inches='tight')
    print(f">> Graphique sauvegardé : disponibilite_par_domaine.png")
    plt.show()
    plt.close()

def generate_summary_report(df, all_domains, domain_counts):
    """Génère un rapport récapitulatif"""
    print("\n" + "="*80)
    print("RAPPORT RECAPITULATIF")
    print("="*80)
    
    urgent_providers = df['Disponibilite'].str.contains('24/7|24h', case=False, na=False).sum()
    defense_providers = df['Description_Service'].str.contains('Défense|défense|militaire', case=False, na=False).sum()
    
    report = f"""
SYNTHESE DE L'ANALYSE DES PRESTATAIRES
{'='*80}

CHIFFRES CLES
  • Total de prestataires analysés : {len(df)}
  • Nombre de domaines d'expertise différents : {len(domain_counts)}
  • Nombre moyen de domaines par prestataire : {df['nb_domaines'].mean():.2f}
  • Types de disponibilité : {df['Disponibilite'].nunique()}

DISPONIBILITE
{df['Disponibilite'].value_counts().to_string()}

DOMAINES LES PLUS REPRESENTES
{domain_counts.head(10).to_string()}

SERVICES SPECIALISES
  • Prestataires 24/7 : {urgent_providers} ({urgent_providers / len(df) * 100:.1f}%)
  • Prestataires spécialisés Défense/Militaire : {defense_providers} ({defense_providers / len(df) * 100:.1f}%)
  • Prestataires multi-domaines (3+) : {len(df[df['nb_domaines'] >= 3])} ({len(df[df['nb_domaines'] >= 3]) / len(df) * 100:.1f}%)

CARACTERISTIQUES DES DESCRIPTIONS
  • Longueur moyenne : {df['description_length'].mean():.0f} caractères
  • Description la plus courte : {df['description_length'].min()} caractères
  • Description la plus longue : {df['description_length'].max()} caractères

INSIGHTS CLES
  1. {urgent_providers / len(df) * 100:.1f}% des prestataires offrent une disponibilité 24/7 pour les urgences
  2. {defense_providers / len(df) * 100:.1f}% sont spécialisés dans les services aux militaires et fonctionnaires
  3. Large couverture des besoins avec {len(domain_counts)} domaines différents représentés
  4. Forte présence de services réactifs et spécialisés mobilité
  
{'='*80}
Fichiers générés dans : {PROCESSED_DATA_DIR}
  >> domaines_expertise.png
  >> disponibilite_prestataires.png
  >> nb_domaines_par_prestataire.png
  >> longueur_descriptions.png
  >> mots_cles_specialises.png
  >> disponibilite_par_domaine.png
  >> rapport_prestataires.txt
{'='*80}
"""
    
    print(report)
    
    # Sauvegarder le rapport
    with open(PROCESSED_DATA_DIR / 'rapport_prestataires.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n>> Rapport sauvegardé : rapport_prestataires.txt")

def main():
    """Fonction principale d'exécution de l'analyse"""
    print("\n" + "="*80)
    print("ANALYSE DES DONNEES - PRESTATAIRES DE SERVICES")
    print("="*80 + "\n")
    
    # Créer le dossier processed s'il n'existe pas
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Charger les données
    df = load_data()
    
    # Exécuter les analyses
    display_basic_info(df)
    all_domains, domain_counts = analyze_expertise_domains(df)
    analyze_availability(df)
    analyze_expertise_count(df)
    analyze_descriptions(df)
    analyze_specialized_services(df)
    analyze_availability_by_domain(df, all_domains)
    generate_summary_report(df, all_domains, domain_counts)
    
    print("\n" + "="*80)
    print("ANALYSE TERMINEE AVEC SUCCES !")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
