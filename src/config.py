"""
Configuration centrale du projet Zeus Armee
Ce fichier contient toutes les variables, clés, chemins de dossiers et fichiers
"""

import os
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Dossiers principaux
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

# Dossiers de données
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Fichiers de données
BESOINS_CSV = RAW_DATA_DIR / "besoins.csv"
BESOINS_V2_CSV = RAW_DATA_DIR / "besoins_v2.csv"  # Nouveaux besoins version 2
BESOINS_ENRICHIS_CSV = RAW_DATA_DIR / "besoins_enrichis.csv"  # Besoins avec entités NER
PRESTATAIRES_CSV = RAW_DATA_DIR / "prestataires.csv"
BESOINS_PROCESSED_CSV = PROCESSED_DATA_DIR / "besoins_processed.csv"
PRESTATAIRES_PROCESSED_CSV = PROCESSED_DATA_DIR / "prestataires_processed.csv"

# Clés API (à ne jamais commiter - utiliser des variables d'environnement)
# API_KEY = os.getenv("API_KEY", "")
# SECRET_KEY = os.getenv("SECRET_KEY", "")

# Paramètres du modèle
MODEL_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
}

# Configuration du matching
MATCHING_CONFIG = {
    "model_name": "dangvantuan/sentence-camembert-large",  # Modèle d'embeddings français
    "top_k": 5,  # Nombre de résultats à retourner par défaut
    "threshold": 0.25,  # Score minimum de similarité (0-1)
    "batch_size": 32,  # Taille des batchs pour l'encodage
    "use_ner": True,  # Activer filtrage et boost NER
    "use_enriched": True,  # Utiliser besoins_enrichis.csv avec entités pré-calculées
}

# Fichiers de résultats
MATCHES_OUTPUT_DIR = PROCESSED_DATA_DIR / "matches"

# Autres configurations
RANDOM_SEED = 42
DEBUG_MODE = True

# Création automatique des dossiers s'ils n'existent pas
def create_directories():
    """Crée tous les dossiers nécessaires s'ils n'existent pas"""
    directories = [
        SRC_DIR,
        DATA_DIR,
        CONFIG_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("Tous les dossiers ont été créés/vérifiés.")

if __name__ == "__main__":
    create_directories()
    print(f"Racine du projet: {PROJECT_ROOT}")
    print(f"Dossier src: {SRC_DIR}")
    print(f"Dossier data: {DATA_DIR}")
    print(f"Dossier raw: {RAW_DATA_DIR}")
    print(f"Dossier processed: {PROCESSED_DATA_DIR}")
