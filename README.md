# Système de Matching Militaire - Zeus Armée

## Description

Système intelligent de mise en relation entre les besoins des militaires et les prestataires de services. Le système combine analyse sémantique (embeddings) et contraintes géographiques pour recommander les prestataires les plus pertinents.

## Architecture Globale

### 1. Données d'Entrée (Raw Data)

**Source :** `data/raw/`
- `besoins.csv` : Besoins des militaires (catégorie, sous-catégorie, description, ville, urgence)
- `besoins_v2.csv` : Dataset étendu avec 50 nouveaux besoins
- `prestataires.csv` : Prestataires de services (entreprise, domaines d'expertise, description, ville, disponibilité)

### 2. Extraction NER (Named Entity Recognition)

**Module :** `src/utils/ner_extractor.py`

Extraction automatique des entités depuis les besoins :
- Ville/Localisation (Nancy, Lyon, Paris...)
- Niveau d'urgence (immédiat, sous 1 mois...)
- Mots-clés métier (famille, véhicule, logement...)

Les villes extraites servent au calcul de distance géographique.

### 3. Génération des Embeddings

**Module :** `src/matching/text_processor.py`  
**Modèle :** `dangvantuan/sentence-camembert-large` (768 dimensions)

#### Texte Client
```
Catégorie: [...] | Type: [...] | Sous-catégorie: [...] | Service: [...] | Description: [...] | Urgence: [...]
```

#### Texte Prestataire
```
Entreprise: [...] | Expertise: [...] | Disponibilité: [...] | Services: [...]
```

**Note importante :** Les villes sont exclues des embeddings pour éviter tout biais géographique. La géographie est traitée séparément.

### 4. Calcul de Similarité Sémantique

**Score d'Embedding (score_emb)** = Similarité cosinus entre vecteurs client et prestataire

Valeur entre 0 et 1 (1 = identique sémantiquement)

### 5. Calcul du Score Géographique

**Formule :** `score_geo = exp(-alpha × distance)`

Où :
- `distance` = Distance en km entre ville du client (NER) et ville du prestataire
- `alpha` = Facteur de décroissance dépendant de l'impact géographique

**Impact Géographique** (défini par mapping `(Catégorie_Majeure, Sous_Categorie)`) :
- **0** : Services en ligne (banque, administration) - distance ignorée
- **1** : Services locaux (achat véhicule, déménagement) - distance modérée
- **2** : Services urgents/proximité (médecin, garde d'enfant) - distance critique

**Source :** `src/scripts/ajouter_impact_geo.py` (160+ mappings)

### 6. Score Total Pondéré

```
score_final = weight_emb × score_emb + weight_geo × score_geo
```

**Pondérations dynamiques selon Impact_Geo :**
- Impact_Geo = 0 : 100% sémantique, 0% géographique
- Impact_Geo = 1 : 65% sémantique, 35% géographique  
- Impact_Geo = 2 : 45% sémantique, 55% géographique

### 7. Filtrage et Recommandations

**Seuil minimum :** score_final ≥ 0.30

**Résultats :** Top 5 prestataires maximum (classés par score décroissant)

Si aucun prestataire ne dépasse le seuil : "Aucun matching trouvé"

## Modules Principaux

```
src/
├── matching/
│   ├── matcher.py              # Moteur de matching principal
│   ├── text_processor.py       # Formatage pour embeddings
│   └── embedding_model.py      # Modèle sentence-transformers
├── utils/
│   ├── ner_extractor.py        # Extraction d'entités
│   ├── geo_utils.py            # Calcul distances (geopy + fallback)
│   └── data_loader.py          # Chargement CSV
├── scripts/
│   ├── enrichir_besoins_ner.py      # Enrichissement NER besoins
│   └── ajouter_impact_geo.py        # Attribution impact géographique
└── run_matching.py             # Script principal d'exécution
```

## Exécution

```bash
# Activer l'environnement virtuel
.\zeus\Scripts\activate

# Lancer le matching
python .\src\run_matching.py
```

**Options disponibles :**
1. Matching interactif (saisie manuelle)
2. Batch complet (tous les besoins)
3. Affichage des résultats existants
4. Analyse des besoins
5. Analyse des prestataires
6. Matching dataset V2

## Technologies

- **Python 3.11**
- **sentence-transformers** : Embeddings français (CamemBERT)
- **geopy** : Géocodage automatique et calcul de distances
- **pandas** : Manipulation de données CSV
- **scikit-learn** : Calcul de similarité cosinus

## Résultats

Fichiers générés dans `data/processed/` :
- `matching_complet_tous_besoins.csv` : Résultats détaillés avec scores
- `rapport_besoins.txt` / `rapport_prestataires.txt` : Analyses statistiques
