"""Test rapide du filtrage pour Location meublée"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
import pandas as pd
from matching import ProviderMatcher

# Charger prestataires
prov_path = Path('data/raw/prestataires.csv')
providers = pd.read_csv(prov_path, sep=';')
print(f">> {len(providers)} prestataires charges")

# Créer matcher avec nom du modèle
matcher = ProviderMatcher(model_name='dangvantuan/sentence-camembert-large', use_ner=True)
print(">> Matcher cree")

# Charger les prestataires
matcher.load_providers(providers)

# Encoder prestataires
print("Encodage prestataires...")
matcher.encode_providers()
print(">> Embeddings generes")

# Requête de test : Location meublée
test_request = {
    'categorie': 'Logement et Installation',
    'service': 'Location meublee',
    'message': 'Mutation d\'ici 3 semaines sur Lyon, je cherche une location meublée à Lyon pour 1 mois le temps de trouver quelque chose de permanent',
    'ville': 'Lyon',
    'urgence': 'Court terme',
    'disponibilite': 'Semaine uniquement'
}

print("\n" + "="*80)
print("TEST: Location meublée à Lyon")
print("="*80)

# Matching AVEC filtrage domaine
results = matcher.find_matches(
    test_request,
    top_k=5,
    threshold=0.2,
    apply_domain_filter=True
)

print(f"\n>> {len(results)} resultats trouves\n")

if len(results) > 0:
    print("Top resultats:")
    for idx, row in results.iterrows():
        score_pct = row['similarity_score'] * 100
        print(f"  {row['Nom_Entreprise']:30s} | Score: {score_pct:.2f}% | Domaines: {row['Domaines_Expertise']}")
else:
    print("Aucun resultat")

print("\n" + "="*80)
print("Verification: L'electricien devrait etre EXCLU!")
print("="*80)
