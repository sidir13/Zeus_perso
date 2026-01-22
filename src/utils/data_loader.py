"""
Utilitaires pour charger les données CSV
"""

import pandas as pd
from pathlib import Path


def load_providers(csv_path):
    """
    Charge le fichier CSV des prestataires
    
    Args:
        csv_path: Chemin vers le fichier prestataires.csv
    
    Returns:
        pandas.DataFrame: DataFrame contenant les prestataires
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas")
    
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    print(f">> {len(df)} prestataires chargés depuis {csv_path.name}")
    
    return df


def load_needs(csv_path):
    """
    Charge le fichier CSV des besoins
    
    Args:
        csv_path: Chemin vers le fichier besoins.csv
    
    Returns:
        pandas.DataFrame: DataFrame contenant les besoins
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas")
    
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    print(f">> {len(df)} besoins chargés depuis {csv_path.name}")
    
    return df


def save_matches(matches_df, output_path):
    """
    Sauvegarde les résultats de matching dans un fichier CSV
    
    Args:
        matches_df: DataFrame contenant les matches
        output_path: Chemin de sortie pour le fichier CSV
    """
    output_path = Path(output_path)
    matches_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
    print(f">> Résultats sauvegardés dans {output_path}")


def validate_providers_df(df):
    """
    Valide que le DataFrame prestataires contient les colonnes nécessaires
    
    Args:
        df: DataFrame à valider
    
    Raises:
        ValueError: Si des colonnes manquent
    """
    required_columns = ['Nom_Entreprise', 'Domaines_Expertise', 
                       'Disponibilite', 'Description_Service']
    
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing}")


def validate_needs_df(df):
    """
    Valide que le DataFrame besoins contient les colonnes nécessaires
    
    Args:
        df: DataFrame à valider
    
    Raises:
        ValueError: Si des colonnes manquent
    """
    required_columns = ['Categorie_Majeure', 'Sous_Categorie', 
                       'Message_Utilisateur', 'Niveau_Urgence']
    
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing}")
