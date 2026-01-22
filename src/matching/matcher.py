"""
Classe principale pour matcher clients et prestataires via embeddings
avec prise en compte des entités NER (ville, temporalité, urgence)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from models.embedding_model import EmbeddingModel
from matching.text_processor import create_provider_text, create_client_request_text, create_need_text
from utils.ner_extractor import NERExtractor


class ProviderMatcher:
    """
    Classe pour faire le matching entre besoins clients et prestataires
    en utilisant des embeddings sémantiques + filtrage NER
    """
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', use_ner=True):
        """
        Initialise le matcher avec un modèle d'embeddings
        
        Args:
            model_name: Nom du modèle Sentence-Transformers à utiliser
            use_ner: Activer le filtrage et boost NER (ville, disponibilité)
        """
        self.embedding_model = EmbeddingModel(model_name)
        self.providers_df = None
        self.provider_embeddings = None
        self.use_ner = use_ner
        self.ner_extractor = NERExtractor() if use_ner else None
    
    def load_providers(self, providers_df):
        """
        Charge les données des prestataires
        
        Args:
            providers_df: DataFrame contenant les prestataires
        """
        self.providers_df = providers_df.copy()
        print(f">> {len(self.providers_df)} prestataires chargés dans le matcher")
    
    def encode_providers(self, show_progress=True):
        """
        Génère les embeddings pour tous les prestataires
        
        Args:
            show_progress: Afficher la barre de progression
        
        Returns:
            numpy.ndarray: Matrice des embeddings
        """
        if self.providers_df is None:
            raise ValueError("Vous devez d'abord charger les prestataires avec load_providers()")
        
        print("\nGénération des embeddings des prestataires...")
        
        # Créer les textes descriptifs
        provider_texts = self.providers_df.apply(create_provider_text, axis=1).tolist()
        
        # Générer les embeddings
        self.provider_embeddings = self.embedding_model.encode(
            provider_texts, 
            show_progress=show_progress
        )
        
        print(f">> Embeddings créés: shape {self.provider_embeddings.shape}")
        return self.provider_embeddings
    
    def find_matches(self, client_request, top_k=5, threshold=0.3, ner_entities=None, impact_geo=1, apply_domain_filter=True):
        """
        Trouve les meilleurs prestataires pour une demande client
        
        Args:
            client_request: Texte ou dict de la demande client
            top_k: Nombre de résultats à retourner
            threshold: Score minimum de similarité (0-1)
            ner_entities: Entités NER extraites (optionnel, calculées auto si None)
            impact_geo: Niveau d'impact géographique (0, 1, 2) - par défaut 1
            apply_domain_filter: Filtrer par domaines d'expertise (True par défaut)
        
        Returns:
            pandas.DataFrame: Meilleurs prestataires avec leurs scores
        """
        if self.provider_embeddings is None:
            raise ValueError("Vous devez d'abord générer les embeddings avec encode_providers()")
        
        # Créer le texte de la requête
        request_text = create_client_request_text(client_request)
        
        # Extraire catégorie et sous-catégorie si disponible
        categorie = None
        sous_categorie = None
        if isinstance(client_request, dict):
            categorie = client_request.get('categorie') or client_request.get('Categorie_Majeure')
            sous_categorie = client_request.get('service') or client_request.get('Sous_Categorie')
        
        # Filtrage préalable par domaines d'expertise
        if apply_domain_filter and (categorie or sous_categorie):
            print(f">> Filtrage active - Categorie: '{categorie}', Sous-categorie: '{sous_categorie}'")
            filtered_df = self._filter_by_domain_relevance(categorie, sous_categorie)
            
            if len(filtered_df) == 0:
                print("Attention: Aucun prestataire ne correspond aux domaines. Desactivation du filtre.")
                filtered_df = self.providers_df.copy()
            else:
                print(f">> Filtrage domaines: {len(self.providers_df)} -> {len(filtered_df)} prestataires")
                # Afficher quelques prestataires retenus
                print(f"  Prestataires retenus: {', '.join(filtered_df['Nom_Entreprise'].head(3).tolist())}...")
            
            # Récupérer les indices filtrés pour les embeddings
            filtered_indices = filtered_df.index.tolist()
            filtered_embeddings = self.provider_embeddings[filtered_indices]
        else:
            filtered_df = self.providers_df.copy()
            filtered_embeddings = self.provider_embeddings
        
        # Extraire entités NER si besoin
        if self.use_ner and ner_entities is None:
            if isinstance(client_request, str):
                ner_entities = self.ner_extractor.extract_all(client_request)
            elif isinstance(client_request, dict):
                message = client_request.get('message', '') or client_request.get('description', '')
                ner_entities = self.ner_extractor.extract_all(message)
        
        # Générer l'embedding de la requête
        request_embedding = self.embedding_model.encode([request_text], show_progress=False)
        
        # Calculer les similarités cosinus sur prestataires filtrés
        similarities = cosine_similarity(request_embedding, filtered_embeddings)[0]
        
        # Ajouter les scores au DataFrame filtré
        results = filtered_df.copy()
        results['similarity_score_base'] = similarities
        results['similarity_score'] = similarities.copy()
        
        # Appliquer filtrage et boosts NER si activé
        if self.use_ner and ner_entities:
            results = self._apply_ner_filtering(results, ner_entities)
            results = self._apply_adaptive_scoring(results, ner_entities, impact_geo)
            results = self._apply_urgency_boost(results, ner_entities)
        
        # Pénaliser prestataires trop génériques (amélioration #3)
        results = self._penalize_generic_providers(results)
        
        # NOUVEAU: Amplifier l'écart entre bons et mauvais matches
        results = self._amplify_score_gap(results)
        
        # NOUVEAU: Filtrer les rangs secondaires avec écart minimal (amélioration #4)
        results = self._filter_secondary_ranks(results)
        
        # Filtrer par seuil minimum (éviter matches faibles < 10%)
        min_score = max(threshold, 0.10)
        results = results[results['similarity_score'] >= min_score]
        
        # NOUVEAU: Limite adaptative selon qualité (max 3) (amélioration #5)
        results = self._apply_adaptive_top_k(results, max_k=min(top_k, 3))
        
        # NOUVEAU: Ajouter labels de confiance (amélioration #6)
        if len(results) > 0:
            results = self._add_confidence_labels(results)
        
        return results
    
    def _filter_by_domain_relevance(self, categorie, sous_categorie):
        """
        Filtre les prestataires dont les domaines d'expertise correspondent 
        sémantiquement à la catégorie et sous-catégorie du besoin
        
        AMÉLIORATION: Filtrage strict avec exclusions explicites
        
        Args:
            categorie: Catégorie majeure du besoin (ex: "Logement et Installation")
            sous_categorie: Sous-catégorie du besoin (ex: "Location meublée")
            
        Returns:
            DataFrame filtré avec prestataires pertinents
        """
        # ÉTAPE 1: EXCLUSIONS STRICTES (éliminer domaines incompatibles)
        # Utiliser des PRÉFIXES courts pour matcher toutes les variantes
        INCOMPATIBLE_DOMAINS = {
            # Pour LOGEMENT: exclure travaux/réparations (préfixes plus courts)
            'location': ['électri', 'électro', 'plomb', 'garage', 'mécan', 'contrôle', 'véhicule', 'auto', 'dépann', 'répara', 'travaux'],
            'colocation': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'auto', 'travaux', 'dépann', 'répara'],
            'logement': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'auto', 'mécan', 'dépann', 'répara'],
            'meublée': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'auto', 'dépann', 'travaux'],
            'immobilier': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'auto', 'dépann', 'répara'],
            
            # Pour FAMILLE: exclure technique
            'garde': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'auto', 'travaux', 'construction', 'dépann', 'répara'],
            'enfant': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'auto', 'travaux', 'dépann', 'répara'],
            'scolarité': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'auto', 'travaux', 'stockage', 'entrepo'],
            'école': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'auto', 'stockage', 'travaux'],
            
            # Pour VÉHICULE: exclure logement/famille
            'véhicule': ['logement', 'location', 'colocation', 'garde', 'enfant', 'crèche', 'nounou', 'stockage', 'immobilier'],
            'auto': ['logement', 'location', 'colocation', 'garde', 'enfant', 'crèche', 'stockage', 'immobilier'],
            'panne': ['logement', 'location', 'garde', 'enfant', 'stockage', 'immobilier'],
            'réparation': ['logement', 'location', 'garde', 'stockage', 'immobilier', 'banque'],
            'garage': ['logement', 'location', 'colocation', 'garde', 'enfant', 'immobilier'],
            
            # Pour BANQUE: exclure technique
            'prêt': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'mécan', 'travaux', 'dépann'],
            'crédit': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'mécan', 'dépann'],
            'banque': ['électri', 'électro', 'plomb', 'garage', 'véhicule', 'travaux', 'dépann'],
            
            # Pour TRAVAUX: exclure services tertiaires
            'plomberie': ['logement', 'location', 'garde', 'enfant', 'banque', 'finance', 'assurance', 'immobilier'],
            'électricité': ['logement', 'location', 'garde', 'enfant', 'banque', 'finance', 'assurance', 'immobilier'],
        }
        
        # Mapping des sous-catégories vers mots-clés OBLIGATOIRES dans les domaines
        REQUIRED_KEYWORDS = {
            # Famille - Garde
            "garde d'enfant": ['garde', 'enfant', 'famille', 'babysitting', 'crèche', 'nounou'],
            "crèche ou nounou": ['garde', 'enfant', 'famille', 'crèche', 'nounou'],
            "scolarité": ['famille', 'scolarité', 'éducation', 'école'],
            "activités périscolaires": ['famille', 'loisirs', 'sport', 'activités', 'enfant'],
            "aide aux devoirs": ['famille', 'éducation', 'soutien', 'scolaire'],
            "garde animaux": ['animaux', 'garde', 'pension', 'chien', 'chat'],
            
            # Travaux et Urgences
            "plomberie urgente": ['plomberie', 'travaux', 'urgence', 'dépannage'],
            "électroménager": ['électroménager', 'réparation', 'dépannage'],
            "réparation urgente": ['réparation', 'dépannage', 'urgence', 'véhicule', 'auto', 'garage'],
            "mise en conformité logement": ['travaux', 'électricité', 'conformité'],
            "rénovation avant vente": ['travaux', 'rénovation'],
            "installation fibre": ['travaux', 'installation', 'internet', 'télécom'],
            
            # Véhicule
            "contrôle technique": ['véhicule', 'auto', 'contrôle', 'technique', 'automobile'],
            "location courte durée": ['location', 'véhicule', 'auto', 'voiture', 'automobile'],
            "achat véhicule": ['véhicule', 'auto', 'vente', 'occasion', 'automobile', 'voiture'],
            "reprogrammation moteur": ['véhicule', 'auto', 'garage', 'mécanique', 'moteur'],
            "réparation urgente": ['garage', 'auto', 'véhicule', 'réparation', 'dépannage', 'mécanique', 'automobile', 'panne'],
            
            # Logement
            "location meublée": ['logement', 'location', 'immobilier', 'appartement', 'meublé', 'habitation'],
            "recherche colocation": ['logement', 'colocation', 'location', 'appartement', 'colocataire'],
            "recherche logement social": ['logement', 'location', 'immobilier', 'social', 'hlm'],
            "déménagement": ['déménagement', 'transport', 'logistique', 'demenage', 'déménageur'],
            "stockage temporaire": ['stockage', 'garde-meuble', 'entreposage', 'box'],
            "état des lieux": ['logement', 'immobilier', 'huissier', 'juridique', 'état', 'constat'],
            "construction maison retraite": ['construction', 'immobilier', 'bâtiment', 'maison', 'promoteur'],
            
            # Banque et Finance
            "prêt immobilier": ['banque', 'finance', 'crédit', 'prêt', 'immobilier'],
            "prêt travaux": ['banque', 'finance', 'crédit', 'prêt'],
            "regroupement crédits": ['banque', 'finance', 'crédit'],
            "placement financier": ['finance', 'banque', 'épargne', 'investissement', 'placement'],
            
            # Assurance
            "assurance habitation": ['assurance', 'habitation', 'logement'],
            "assurance auto jeune conducteur": ['assurance', 'auto', 'véhicule'],
            "mutuelle santé": ['assurance', 'mutuelle', 'santé'],
            "prévoyance": ['assurance', 'prévoyance'],
            
            # Administratif
            "carte grise": ['administratif', 'carte', 'véhicule', 'démarches'],
            "passeport express": ['administratif', 'passeport', 'démarches', 'papiers'],
            "titre de séjour conjoint": ['administratif', 'démarches', 'juridique'],
            "changement situation familiale": ['administratif', 'juridique', 'démarches'],
            
            # Santé
            "dentiste d'urgence": ['santé', 'dentiste', 'dentaire', 'urgence'],
            "kiné urgence": ['santé', 'kiné', 'kinésithérapie', 'rééducation'],
            "ophtalmologue": ['santé', 'ophtalmologue', 'vision', 'lunettes'],
            "accompagnement familial": ['santé', 'psychologue', 'accompagnement', 'famille'],
            "gestion stress opérationnel": ['santé', 'psychologue', 'stress', 'accompagnement'],
            
            # Emploi et Formation
            "recherche emploi conjoint": ['emploi', 'travail', 'recrutement', 'job'],
            "reconversion professionnelle": ['formation', 'reconversion', 'emploi'],
            "bilan de compétences": ['emploi', 'formation', 'bilan', 'orientation'],
            "aide à la création entreprise": ['entreprise', 'création', 'conseil', 'accompagnement'],
            "préparation retraite": ['retraite', 'conseil', 'finance', 'accompagnement'],
            "permis poids lourd": ['formation', 'permis', 'conduite'],
            "langue étrangère": ['formation', 'langue', 'cours', 'apprentissage'],
            
            # Services Express
            "transport express": ['transport', 'livraison', 'coursier', 'urgence'],
            "coiffure": ['coiffure', 'beauté', 'esthétique'],
            "pressing express": ['pressing', 'nettoyage', 'blanchisserie'],
        }
        
        # Normaliser sous-catégorie (minuscules + supprimer accents)
        import unicodedata
        sous_cat_lower = sous_categorie.lower() if sous_categorie else ""
        sous_cat_normalized = unicodedata.normalize('NFD', sous_cat_lower)
        sous_cat_normalized = ''.join(char for char in sous_cat_normalized if unicodedata.category(char) != 'Mn')
        
        # ÉTAPE 2: Trouver les mots-clés requis (chercher avec ET sans accents)
        keywords = set()
        for key, values in REQUIRED_KEYWORDS.items():
            key_normalized = unicodedata.normalize('NFD', key)
            key_normalized = ''.join(char for char in key_normalized if unicodedata.category(char) != 'Mn')
            
            if key_normalized in sous_cat_normalized or sous_cat_normalized in key_normalized:
                keywords.update(values)
                break
        
        # Si aucun mapping trouvé, extraire mots-clés génériques
        if len(keywords) == 0:
            if sous_categorie:
                mots = [m.lower() for m in sous_categorie.split() if len(m) > 4]
                keywords.update(mots)
            if categorie and len(keywords) == 0:
                mots = [m.lower() for m in categorie.split() if len(m) > 4]
                keywords.update(mots)
        
        # Si toujours aucun mot-clé, retourner tous les prestataires
        if len(keywords) == 0:
            return self.providers_df.copy()
        
        # ÉTAPE 3: Identifier les exclusions pour cette catégorie (aussi normalisé)
        exclusions = set()
        for key, excluded_words in INCOMPATIBLE_DOMAINS.items():
            key_norm = unicodedata.normalize('NFD', key)
            key_norm = ''.join(char for char in key_norm if unicodedata.category(char) != 'Mn')
            
            if key_norm in sous_cat_normalized:
                exclusions.update(excluded_words)
        
        # ÉTAPE 4: Déterminer si matching strict (multi-mots) requis
        # Certaines catégories nécessitent PLUSIEURS mots-clés pour éviter les faux positifs
        STRICT_CATEGORIES = ['location', 'logement', 'colocation', 'scolarite', 'ecole', 'pret', 'credit', 'banque']
        require_multiple_keywords = any(cat in sous_cat_normalized for cat in STRICT_CATEGORIES)
        min_keyword_matches = 2 if require_multiple_keywords else 1
        
        # ÉTAPE 5: Filtrer avec logique stricte
        def is_domain_compatible(domaines_str):
            if pd.isna(domaines_str):
                return False
            
            domaines_lower = domaines_str.lower()
            
            # ÉTAPE 5.1: Vérifier exclusions (bloquant)
            for exclusion in exclusions:
                if exclusion in domaines_lower:
                    return False  # EXCLU
            
            # ÉTAPE 5.2: Compter les mots-clés présents
            matched_keywords = sum(1 for keyword in keywords if keyword in domaines_lower)
            
            # ÉTAPE 5.3: Valider selon le seuil requis
            return matched_keywords >= min_keyword_matches
        
        filtered_df = self.providers_df[
            self.providers_df['Domaines_Expertise'].apply(is_domain_compatible)
        ].copy()
        
        return filtered_df
    
    def _apply_ner_filtering(self, results, ner_entities):
        """
        Filtre les prestataires selon les contraintes NER
        
        Args:
            results: DataFrame des prestataires avec scores
            ner_entities: Entités NER extraites
            
        Returns:
            DataFrame filtré
        """
        contraintes = ner_entities.get('contraintes_matching', {})
        contrainte_dispo = contraintes.get('disponibilite', 'ALL')
        
        # Filtrage par disponibilité
        if contrainte_dispo != 'ALL' and 'Disponibilite' in results.columns:
            mask_compatible = results['Disponibilite'].apply(
                lambda x: self.ner_extractor.is_compatible_disponibilite(x, contrainte_dispo)
            )
            results = results[mask_compatible]
        
        return results
    
    def _apply_adaptive_scoring(self, results, ner_entities, impact_geo):
        """
        AMÉLIORATION #4: Pondération adaptive selon impact_geo
        
        Au lieu de multiplication pure (score_base × score_geo), utilise
        une combinaison pondérée qui varie selon le type de service
        
        Args:
            results: DataFrame des prestataires
            ner_entities: Entités NER extraites
            impact_geo: Niveau d'impact géographique (0, 1, 2)
            
        Returns:
            DataFrame avec scores pondérés adaptivement
        """
        ville_besoin = ner_entities.get('ville_detectee')
        
        # Poids selon impact_geo
        ADAPTIVE_WEIGHTS = {
            0: {'semantic': 1.0, 'geo': 0.0},    # Services en ligne: 100% sémantique
            1: {'semantic': 0.65, 'geo': 0.35},  # Services locaux: 65% sém + 35% geo
            2: {'semantic': 0.45, 'geo': 0.55},  # Urgences: 45% sém + 55% geo
        }
        
        weights = ADAPTIVE_WEIGHTS.get(impact_geo, {'semantic': 0.7, 'geo': 0.3})
        
        if 'Ville' in results.columns and ville_besoin:
            # Calculer le score géographique
            geo_scores = results['Ville'].apply(
                lambda x: self.ner_extractor.calculate_geo_score(
                    ville_besoin, x, impact_geo
                )
            )
            results['geo_score'] = geo_scores
            
            # Combinaison pondérée (au lieu de multiplication)
            results['similarity_score'] = (
                weights['semantic'] * results['similarity_score_base'] +
                weights['geo'] * geo_scores
            )
        else:
            # Pas de ville ou pas de colonne Ville: score = sémantique pur
            results['geo_score'] = 1.0
            results['similarity_score'] = results['similarity_score_base']
        
        return results
    
    def _filter_secondary_ranks(self, results):
        """
        Filtre les rangs secondaires selon écart avec le TOP 1
        
        RÈGLE: Un résultat peut être affiché seulement si:
        - score >= 70% du score TOP 1 (écart max 30%)
        - score >= 0.30 absolu (seuil plancher)
        
        Args:
            results: DataFrame avec les scores
            
        Returns:
            DataFrame filtré
        """
        if len(results) == 0:
            return results
        
        # Trier d'abord par score
        results_sorted = results.sort_values('similarity_score', ascending=False)
        
        # Score du TOP 1
        top_score = results_sorted.iloc[0]['similarity_score']
        min_score_relative = top_score * 0.70  # Écart max 30%
        min_score_absolute = 0.30  # Seuil plancher
        
        # Filtrer
        filtered = results_sorted[
            (results_sorted['similarity_score'] >= min_score_relative) &
            (results_sorted['similarity_score'] >= min_score_absolute)
        ]
        
        return filtered
    
    def _apply_adaptive_top_k(self, results, max_k=3):
        """
        Ajuste dynamiquement le nombre de résultats selon qualité du TOP 1
        
        RÈGLE:
        - TOP 1 >= 85%: afficher 3 résultats (proposer alternatives)
        - TOP 1 >= 70%: afficher 2 résultats (une alternative)
        - TOP 1 >= 50%: afficher 1 seul résultat
        - TOP 1 < 50%: afficher 1 résultat avec disclaimer
        
        Args:
            results: DataFrame trié par score
            max_k: Limite maximale (3 par défaut)
            
        Returns:
            DataFrame limité au nombre optimal
        """
        if len(results) == 0:
            return results
        
        top_score = results.iloc[0]['similarity_score']
        
        # Déterminer le nombre optimal de résultats
        if top_score >= 0.85:
            optimal_k = 3  # Excellent match: proposer alternatives
        elif top_score >= 0.70:
            optimal_k = 2  # Bon match: une alternative
        elif top_score >= 0.50:
            optimal_k = 1  # Match moyen: limiter au meilleur
        else:
            optimal_k = 1  # Match faible: un seul avec prudence
        
        return results.head(min(optimal_k, max_k))
    
    def _add_confidence_labels(self, results):
        """
        Ajoute un label de confiance pour chaque résultat
        
        Labels:
        - >= 85%: "Tres pertinent"
        - >= 70%: "Pertinent"
        - >= 50%: "Approchant"
        - < 50%: "A verifier"
        
        Args:
            results: DataFrame avec les scores
            
        Returns:
            DataFrame avec colonne 'confidence' ajoutée
        """
        def get_confidence_label(score):
            if score >= 0.85:
                return "Tres pertinent"
            elif score >= 0.70:
                return "Pertinent"
            elif score >= 0.50:
                return "Approchant"
            else:
                return "A verifier"
        
        results['confidence'] = results['similarity_score'].apply(get_confidence_label)
        return results
    
    def _apply_urgency_boost(self, results, ner_entities):
        """
        AMÉLIORATION #2: Boost pour prestataires 24/7 quand besoin URGENT
        
        Augmente le score final de 15% quand:
        - Besoin = IMMEDIATE
        - Prestataire = 24/7 ou "urgence"
        
        Args:
            results: DataFrame des prestataires
            ner_entities: Entités NER extraites
            
        Returns:
            DataFrame avec boost urgence appliqué
        """
        urgence = ner_entities.get('urgence_deduite', 'STANDARD')
        
        if urgence == 'IMMEDIATE' and 'Disponibilite' in results.columns:
            def calc_urgency_boost(dispo):
                if pd.isna(dispo):
                    return 1.0
                dispo_lower = dispo.lower()
                if '24/7' in dispo_lower or 'urgence' in dispo_lower:
                    return 1.15  # +15% pour prestataires urgence
                return 1.0
            
            results['urgency_boost'] = results['Disponibilite'].apply(calc_urgency_boost)
            results['similarity_score'] *= results['urgency_boost']
            
            # S'assurer de rester dans [0, 1]
            results['similarity_score'] = results['similarity_score'].clip(upper=1.0)
        
        return results
    
    def _penalize_generic_providers(self, results):
        """
        AMÉLIORATION #3: Pénalise prestataires trop généralistes
        
        Réduit le score des prestataires avec beaucoup de domaines
        (moins spécialisés, potentiellement moins experts)
        
        Args:
            results: DataFrame des prestataires
            
        Returns:
            DataFrame avec pénalité générique appliquée
        """
        def calc_specialization_penalty(domaines_str):
            if pd.isna(domaines_str):
                return 0.95  # -5% si pas de domaines définis
            
            # Compter le nombre de domaines (séparés par virgules)
            nb_domaines = len([d.strip() for d in domaines_str.split(',') if d.strip()])
            
            if nb_domaines >= 6:
                return 0.85  # -15% si >= 6 domaines (trop généraliste)
            elif nb_domaines >= 5:
                return 0.90  # -10% si 5 domaines
            elif nb_domaines >= 4:
                return 0.95  # -5% si 4 domaines
            else:
                return 1.0   # Pas de pénalité si <= 3 domaines (spécialisé)
        
        if 'Domaines_Expertise' in results.columns:
            results['specialization_factor'] = results['Domaines_Expertise'].apply(
                calc_specialization_penalty
            )
            results['similarity_score'] *= results['specialization_factor']
        
        return results
    
    def _amplify_score_gap(self, results):
        """
        Amplifie l'écart entre bons et mauvais matches
        
        Stratégie de discrimination :
        - Scores >= 0.70 : BOOST +20% à +25% (excellents matches)
        - Scores 0.60-0.70 : BOOST +15% (très bons matches)
        - Scores 0.50-0.60 : BOOST +10% (bons matches)
        - Scores 0.45-0.50 : BOOST +5% (corrects)
        - Scores 0.35-0.45 : Neutre (moyens)
        - Scores 0.30-0.35 : PÉNALITÉ -15% (faibles)
        - Scores < 0.30 : PÉNALITÉ -30% (très faibles)
        
        Objectif : Créer un écart marqué pour faciliter la décision
        
        Args:
            results: DataFrame avec similarity_score
            
        Returns:
            DataFrame avec scores amplifiés
        """
        if len(results) == 0:
            return results
        
        # Sauvegarder le score original
        results['similarity_score_before_amplify'] = results['similarity_score'].copy()
        
        def amplify_score(score):
            """Amplifie l'écart : boost les bons, pénalise les mauvais"""
            if score >= 0.70:
                # Excellents : boost +25%
                return min(score * 1.25, 1.0)
            elif score >= 0.60:
                # Très bons : boost +15%
                return score * 1.15
            elif score >= 0.50:
                # Bons : boost +10%
                return score * 1.10
            elif score >= 0.45:
                # Corrects : boost +5%
                return score * 1.05
            elif score >= 0.35:
                # Moyens : neutre
                return score
            elif score >= 0.30:
                # Faibles : pénalité -15%
                return score * 0.85
            else:
                # Très faibles : pénalité -30%
                return score * 0.70
        
        # Appliquer l'amplification
        results['similarity_score'] = results['similarity_score'].apply(amplify_score)
        
        # Garantir bornes [0, 1]
        results['similarity_score'] = results['similarity_score'].clip(0, 1)
        
        return results
    
    def _apply_ner_geo_score(self, results, ner_entities, impact_geo):
        """
        Applique le score géographique rigoureux basé sur la distance réelle
        
        Version 2.0: Score mathématiquement borné dans [0,1] par construction
        Formule: score_geo = exp(-alpha * distance) où alpha dépend de impact_geo
        
        Args:
            results: DataFrame des prestataires
            ner_entities: Entités NER extraites
            impact_geo: Niveau d'impact géographique (0=nul, 1=modéré, 2=fort)
            
        Returns:
            DataFrame avec scores géographiques appliqués
        """
        ville_besoin = ner_entities.get('ville_detectee')
        
        if 'Ville' in results.columns:
            # Calculer le score géographique pour chaque prestataire
            geo_scores = results['Ville'].apply(
                lambda x: self.ner_extractor.calculate_geo_score(
                    ville_besoin, x, impact_geo
                )
            )
            
            # Stocker le score géographique
            results['geo_score'] = geo_scores
            
            # Score final = score sémantique × score géographique
            # Les deux sont dans [0,1] donc le produit reste dans [0,1]
            results['similarity_score'] = results['similarity_score_base'] * geo_scores
            
            # Pas de normalisation nécessaire: le produit de deux nombres dans [0,1]
            # est mathématiquement garanti d'être dans [0,1]
        
        return results
    
    def _apply_ner_boost(self, results, ner_entities):
        """
        ANCIENNE VERSION - Conservée pour compatibilité ascendante
        
        ⚠️ DEPRECATED: Utilisez _apply_ner_geo_score() à la place
        
        Applique des boosts de score selon correspondance géographique
        Utilise l'ancienne méthode avec multiplicateurs arbitraires
        
        Args:
            results: DataFrame des prestataires
            ner_entities: Entités NER extraites
            
        Returns:
            DataFrame avec scores boostés
        """
        ville_besoin = ner_entities.get('ville_detectee')
        contraintes = ner_entities.get('contraintes_matching', {})
        contrainte_ville = contraintes.get('ville', 'FLEXIBLE')
        
        if ville_besoin and 'Ville' in results.columns:
            # Calculer boost géographique pour chaque prestataire
            boosts = results['Ville'].apply(
                lambda x: self.ner_extractor.calculate_geo_boost(
                    ville_besoin, x, contrainte_ville
                )
            )
            
            # Appliquer le boost au score de similarité
            results['geo_boost'] = boosts
            results['similarity_score'] = results['similarity_score_base'] * boosts
            
            # Recalculer pour rester dans [0, 1]
            max_score = results['similarity_score'].max()
            if max_score > 1.0:
                results['similarity_score'] = results['similarity_score'] / max_score
        
        return results
        results = results.sort_values('similarity_score', ascending=False)
        results = results.head(top_k)
        
        return results
    
    def match_need_row(self, need_row, top_k=3, threshold=0.3, apply_domain_filter=True):
        """
        Trouve les prestataires pour un besoin du CSV
        
        Args:
            need_row: pandas.Series représentant une ligne du CSV besoins
            top_k: Nombre de résultats
            threshold: Score minimum
            apply_domain_filter: Filtrer par domaines (True par défaut)
        
        Returns:
            pandas.DataFrame: Meilleurs prestataires
        """
        # Créer un dict complet pour find_matches (pas juste le texte)
        need_dict = {
            'message': need_row.get('Message_Utilisateur', ''),
            'categorie': need_row.get('Categorie_Majeure'),
            'service': need_row.get('Sous_Categorie'),
            'urgence': need_row.get('Niveau_Urgence'),
            'ville': need_row.get('Ville_Detectee') if 'Ville_Detectee' in need_row else None,
            'disponibilite': need_row.get('Contrainte_Disponibilite') if 'Contrainte_Disponibilite' in need_row else None
        }
        
        # Récupérer impact_geo depuis la colonne si disponible
        impact_geo = int(need_row.get('Impact_Geo', 1))  # Par défaut 1 si absent
        
        return self.find_matches(
            need_dict,  # Passer le dict complet, pas juste le texte
            top_k=top_k, 
            threshold=threshold, 
            impact_geo=impact_geo,
            apply_domain_filter=apply_domain_filter
        )
    
    def batch_match(self, needs_df, top_k=3, threshold=0.3):
        """
        Fait le matching pour plusieurs besoins en batch
        
        Args:
            needs_df: DataFrame contenant les besoins
            top_k: Nombre de résultats par besoin
            threshold: Score minimum
        
        Returns:
            list: Liste de tuples (index_besoin, DataFrame_matches)
        """
        results = []
        
        print(f"\nMatching en batch pour {len(needs_df)} besoins...")
        
        for idx, need_row in needs_df.iterrows():
            matches = self.match_need_row(need_row, top_k=top_k, threshold=threshold)
            results.append((idx, matches))
        
        print(f">> Batch matching terminé: {len(results)} besoins traités")
        return results
    
    def get_model_info(self):
        """Retourne les informations sur le modèle utilisé"""
        return self.embedding_model.get_model_info()
