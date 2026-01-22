"""
Module NER (Named Entity Recognition) pour extraction d'entités clés
depuis les messages utilisateurs

Version 2.0: Score géographique mathématiquement rigoureux
- Fonction exponentielle naturellement bornée dans [0,1]
- Impact modulé par impact_geo (0, 1, 2)
- Calcul de distance GPS réel via geo_utils
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from math import exp
from .geo_utils import get_distance_entre_villes


class NERExtractor:
    """
    Extracteur d'entités nommées pour le matching de services
    Extrait: Ville, Temporalité, Urgence
    """
    
    # Villes françaises courantes dans le contexte militaire
    VILLES_FRANCE = [
        'Paris', 'Lyon', 'Marseille', 'Toulouse', 'Lille', 'Bordeaux',
        'Nice', 'Nantes', 'Strasbourg', 'Montpellier', 'Rennes', 'Toulon',
        'Grenoble', 'Dijon', 'Angers', 'Brest', 'Le Mans', 'Metz',
        'Reims', 'Orléans', 'Bourges', 'Vendée', 'Versailles', 'Rouen',
        'Mulhouse', 'Caen', 'Nancy', 'Saint-Étienne', 'Avignon'
    ]
    
    # Patterns temporels
    PATTERNS_IMMEDIATE = [
        r'demain', r"aujourd'hui", r'ce soir', r'tout de suite',
        r'immédiat', r'urgent', r'dans \d{1,2}h', r'sous \d{1,2}h',
        r'dans 24h', r'dans 48h', r'après-demain'
    ]
    
    PATTERNS_SHORT_TERM = [
        r'dans \d+ jours?', r'dans \d+ semaines?', r"d'ici \d+ jours?",
        r"d'ici \d+ semaines?", r'la semaine prochaine', r'le mois prochain',
        r'court terme', r'prochainement'
    ]
    
    PATTERNS_PLANNED = [
        r'dans \d+ mois', r'en \w+', r'pour \w+ \d{4}',
        r'planifié', r'prévu', r'programmé', r'dans \d+ ans?'
    ]
    
    # Patterns d'urgence implicite
    KEYWORDS_URGENCE_HIGH = [
        'urgence', 'urgent', 'immédiat', 'critique', 'panne',
        'fuite', 'cassé', 'bloqué', 'rage de dent', 'douleur',
        'mission imprévue', 'imprévu', 'dernière minute'
    ]
    
    KEYWORDS_URGENCE_MEDIUM = [
        'rapidement', 'vite', 'bientôt', 'court terme',
        'sous peu', 'dès que possible'
    ]
    
    def extract_ville(self, message: str) -> Optional[str]:
        """
        Extrait la ville mentionnée dans le message
        
        Args:
            message: Message utilisateur en texte libre
            
        Returns:
            Nom de la ville détectée ou None
        """
        message_lower = message.lower()
        
        # Recherche directe des villes
        for ville in self.VILLES_FRANCE:
            # Patterns: "à Paris", "sur Lyon", "de Marseille", etc.
            patterns = [
                rf'\bà {ville.lower()}\b',
                rf'\bsur {ville.lower()}\b',
                rf'\bde {ville.lower()}\b',
                rf'\bvers {ville.lower()}\b',
                rf'\bpour {ville.lower()}\b',
                rf'\b{ville.lower()}\b'
            ]
            
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return ville
        
        # Recherche indirecte par contexte
        # Mutation sur X, unité à X
        match_mutation = re.search(r'mutation (?:sur|à|vers) (\w+)', message_lower)
        if match_mutation:
            ville_candidate = match_mutation.group(1).capitalize()
            if ville_candidate in self.VILLES_FRANCE:
                return ville_candidate
        
        match_unite = re.search(r'unité (?:de|à) (\w+)', message_lower)
        if match_unite:
            ville_candidate = match_unite.group(1).capitalize()
            if ville_candidate in self.VILLES_FRANCE:
                return ville_candidate
        
        return None
    
    def extract_temporalite(self, message: str) -> Dict[str, any]:
        """
        Extrait la temporalité et normalise l'horizon
        
        Args:
            message: Message utilisateur
            
        Returns:
            Dict avec date_detectee, horizon_temporel, jours_estimation
        """
        message_lower = message.lower()
        result = {
            'date_detectee': None,
            'horizon_temporel': None,
            'jours_estimation': None
        }
        
        # Détection IMMEDIATE (≤ 24h)
        for pattern in self.PATTERNS_IMMEDIATE:
            if re.search(pattern, message_lower):
                result['horizon_temporel'] = 'IMMEDIATE'
                result['jours_estimation'] = 0
                
                # Extraction date précise si possible
                if 'demain' in message_lower:
                    result['date_detectee'] = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                elif "aujourd'hui" in message_lower or 'ce soir' in message_lower:
                    result['date_detectee'] = datetime.now().strftime('%Y-%m-%d')
                elif 'après-demain' in message_lower:
                    result['date_detectee'] = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
                
                return result
        
        # Détection SHORT_TERM (≤ 30 jours)
        for pattern in self.PATTERNS_SHORT_TERM:
            match = re.search(pattern, message_lower)
            if match:
                result['horizon_temporel'] = 'SHORT_TERM'
                
                # Extraction du nombre de jours/semaines
                nb_match = re.search(r'(\d+)\s+(jour|semaine)', match.group())
                if nb_match:
                    nb = int(nb_match.group(1))
                    unite = nb_match.group(2)
                    
                    if unite == 'jour':
                        result['jours_estimation'] = nb
                    elif unite == 'semaine':
                        result['jours_estimation'] = nb * 7
                    
                    result['date_detectee'] = (datetime.now() + timedelta(days=result['jours_estimation'])).strftime('%Y-%m-%d')
                else:
                    result['jours_estimation'] = 15  # Estimation moyenne
                
                return result
        
        # Détection PLANNED (> 30 jours)
        for pattern in self.PATTERNS_PLANNED:
            match = re.search(pattern, message_lower)
            if match:
                result['horizon_temporel'] = 'PLANNED'
                
                # Extraction mois
                nb_mois_match = re.search(r'(\d+)\s+mois', match.group())
                if nb_mois_match:
                    nb_mois = int(nb_mois_match.group(1))
                    result['jours_estimation'] = nb_mois * 30
                    result['date_detectee'] = (datetime.now() + timedelta(days=result['jours_estimation'])).strftime('%Y-%m-%d')
                else:
                    result['jours_estimation'] = 90  # Estimation par défaut
                
                # Détection de mois spécifique (septembre, mars, etc.)
                mois_patterns = {
                    'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4,
                    'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
                    'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
                }
                
                for mois_nom, mois_num in mois_patterns.items():
                    if mois_nom in message_lower:
                        # Calculer la date approximative
                        current_date = datetime.now()
                        target_month = mois_num
                        target_year = current_date.year
                        
                        # Si le mois est passé, prendre l'année suivante
                        if target_month < current_date.month:
                            target_year += 1
                        
                        try:
                            target_date = datetime(target_year, target_month, 15)  # Milieu du mois
                            result['date_detectee'] = target_date.strftime('%Y-%m-%d')
                            result['jours_estimation'] = (target_date - current_date).days
                        except:
                            pass
                        
                        break
                
                return result
        
        # Si aucune temporalité explicite, déduire du contexte
        if any(kw in message_lower for kw in self.KEYWORDS_URGENCE_HIGH):
            result['horizon_temporel'] = 'IMMEDIATE'
            result['jours_estimation'] = 0
        elif any(kw in message_lower for kw in self.KEYWORDS_URGENCE_MEDIUM):
            result['horizon_temporel'] = 'SHORT_TERM'
            result['jours_estimation'] = 7
        
        return result
    
    def extract_urgence(self, message: str, niveau_urgence_col: Optional[str] = None) -> str:
        """
        Déduit le niveau d'urgence du message
        
        Args:
            message: Message utilisateur
            niveau_urgence_col: Niveau d'urgence depuis la colonne CSV (si disponible)
            
        Returns:
            Niveau d'urgence: IMMEDIATE, SHORT_TERM, STANDARD, PLANNED
        """
        # Si niveau d'urgence fourni dans les données, le normaliser
        if niveau_urgence_col:
            niveau_lower = niveau_urgence_col.lower()
            if 'immédiat' in niveau_lower or 'urgent' in niveau_lower:
                return 'IMMEDIATE'
            elif 'court terme' in niveau_lower:
                return 'SHORT_TERM'
            elif 'planifié' in niveau_lower:
                return 'PLANNED'
            else:
                return 'STANDARD'
        
        # Sinon, déduire du message
        message_lower = message.lower()
        
        # Urgence haute
        if any(kw in message_lower for kw in self.KEYWORDS_URGENCE_HIGH):
            return 'IMMEDIATE'
        
        # Urgence moyenne
        if any(kw in message_lower for kw in self.KEYWORDS_URGENCE_MEDIUM):
            return 'SHORT_TERM'
        
        # Recherche patterns temporels
        tempo_info = self.extract_temporalite(message)
        if tempo_info['horizon_temporel']:
            return tempo_info['horizon_temporel']
        
        # Par défaut
        return 'STANDARD'
    
    def extract_all(self, message: str, niveau_urgence_col: Optional[str] = None) -> Dict[str, any]:
        """
        Extraction complète de toutes les entités
        
        Args:
            message: Message utilisateur
            niveau_urgence_col: Niveau d'urgence depuis CSV (optionnel)
            
        Returns:
            Dict avec toutes les entités extraites
        """
        ville = self.extract_ville(message)
        tempo = self.extract_temporalite(message)
        urgence = self.extract_urgence(message, niveau_urgence_col)
        
        # Déterminer les contraintes de matching
        contraintes = self._determiner_contraintes(ville, tempo['horizon_temporel'], urgence)
        
        return {
            'ville_detectee': ville,
            'date_detectee': tempo['date_detectee'],
            'horizon_temporel': tempo['horizon_temporel'] or urgence,
            'jours_estimation': tempo['jours_estimation'],
            'urgence_deduite': urgence,
            'contraintes_matching': contraintes
        }
    
    def _determiner_contraintes(self, ville: Optional[str], horizon: Optional[str], urgence: str) -> Dict[str, str]:
        """
        Détermine les contraintes de matching selon les entités extraites
        
        Returns:
            Dict avec contraintes pour ville et disponibilité
        """
        contraintes = {
            'ville': 'FLEXIBLE',  # STRICT, PREFERRED, FLEXIBLE, NATIONAL
            'disponibilite': 'ALL'  # 24/7, RAPIDE, SEMAINE, ALL
        }
        
        # Contrainte ville
        if ville:
            contraintes['ville'] = 'PREFERRED'  # Match ville prioritaire mais pas bloquant
        else:
            contraintes['ville'] = 'NATIONAL'  # Prestataires nationaux/en ligne OK
        
        # Contrainte disponibilité selon urgence
        if urgence == 'IMMEDIATE' or horizon == 'IMMEDIATE':
            contraintes['disponibilite'] = '24/7'
        elif urgence == 'SHORT_TERM' or horizon == 'SHORT_TERM':
            contraintes['disponibilite'] = 'RAPIDE'
        elif urgence == 'PLANNED' or horizon == 'PLANNED':
            contraintes['disponibilite'] = 'ALL'
        else:
            contraintes['disponibilite'] = 'SEMAINE'
        
        return contraintes
    
    def is_compatible_disponibilite(self, prestataire_dispo: str, contrainte_dispo: str) -> bool:
        """
        Vérifie si la disponibilité du prestataire est compatible
        
        Args:
            prestataire_dispo: Disponibilité du prestataire (ex: "24/7", "Semaine uniquement")
            contrainte_dispo: Contrainte déduite (ex: "24/7", "RAPIDE", "SEMAINE", "ALL")
            
        Returns:
            True si compatible, False sinon
        """
        prestataire_lower = prestataire_dispo.lower()
        
        if contrainte_dispo == 'ALL':
            return True
        
        if contrainte_dispo == '24/7':
            return '24/7' in prestataire_lower or 'urgence' in prestataire_lower
        
        if contrainte_dispo == 'RAPIDE':
            return ('24/7' in prestataire_lower or 
                    'urgence' in prestataire_lower or
                    'rapide' in prestataire_lower or
                    'samedi' in prestataire_lower or
                    'en ligne' in prestataire_lower)
        
        if contrainte_dispo == 'SEMAINE':
            return True  # Tous compatibles
        
        return True
    
    def calculate_geo_score(self, ville_besoin: Optional[str], ville_prestataire: str, 
                           impact_geo: int) -> float:
        """
        Calcule un score géographique mathématiquement rigoureux
        
        Formule exponentielle naturellement bornée dans [0, 1]:
            score_geo = exp(-alpha * distance)
        
        Où alpha dépend de impact_geo:
            - impact_geo = 0 : alpha = 0.0 → score = 1.0 constant (services en ligne)
            - impact_geo = 1 : alpha = 0.015 → décroissance modérée (services locaux)
            - impact_geo = 2 : alpha = 0.05 → décroissance forte (urgences, proximité critique)
        
        Propriétés mathématiques:
            - Toujours dans [0, 1] par construction
            - Continue et strictement décroissante
            - Pas de normalisation a posteriori
            - Distance = 0 → score = 1.0
            - Distance → ∞ → score → 0
        
        Args:
            ville_besoin: Ville détectée dans le besoin (peut être None)
            ville_prestataire: Ville du prestataire (str, toujours définie)
            impact_geo: Niveau d'impact géographique (0, 1, ou 2)
            
        Returns:
            Score géographique dans [0, 1]
        """
        # Paramètres de décroissance selon impact_geo
        ALPHA_COEFFICIENTS = {
            0: 0.0,      # Aucun impact → score constant = 1.0
            1: 0.015,    # Impact modéré → 50% à 46km, 10% à 154km
            2: 0.05,     # Impact fort → 50% à 14km, 10% à 46km
        }
        
        # Validation de impact_geo
        if impact_geo not in ALPHA_COEFFICIENTS:
            raise ValueError(f"impact_geo doit être 0, 1 ou 2 (reçu: {impact_geo})")
        
        alpha = ALPHA_COEFFICIENTS[impact_geo]
        
        # Cas 1: impact_geo = 0 → service en ligne, distance non pertinente
        if alpha == 0.0:
            return 1.0
        
        # Cas 2: Pas de ville dans le besoin → score neutre
        if not ville_besoin:
            return 0.8  # Pénalité légère pour incertitude géographique
        
        # Cas 3: Même ville → score maximal
        if ville_besoin.lower().strip() == ville_prestataire.lower().strip():
            return 1.0
        
        # Cas 4: Villes différentes → calcul de distance
        distance_km = get_distance_entre_villes(ville_besoin, ville_prestataire)
        
        if distance_km is None:
            # Ville non reconnue dans la base GPS → pénalité modérée
            return 0.7
        
        # Formule exponentielle décroissante
        score_geo = exp(-alpha * distance_km)
        
        # Garantie mathématique: score toujours dans [0, 1]
        # (la formule garantit cela, mais on ajoute une assertion pour la sécurité)
        assert 0.0 <= score_geo <= 1.0, f"Score hors bornes: {score_geo}"
        
        return score_geo
    
    def calculate_geo_boost(self, ville_besoin: Optional[str], ville_prestataire: str, 
                           contrainte_ville: str) -> float:
        """
        ANCIENNE VERSION - Conservée pour compatibilité ascendante
        
        ⚠️ DEPRECATED: Utilisez calculate_geo_score() à la place
        
        Cette méthode utilise des multiplicateurs arbitraires (0.5 à 1.5)
        qui ne sont pas mathématiquement rigoureux et nécessitent des normalisations.
        """
        if not ville_besoin:
            return 1.0  # Pas de préférence
        
        ville_match = ville_besoin.lower() == ville_prestataire.lower()
        
        if contrainte_ville == 'STRICT':
            return 1.5 if ville_match else 0.5
        elif contrainte_ville == 'PREFERRED':
            return 1.3 if ville_match else 0.9
        elif contrainte_ville == 'FLEXIBLE':
            return 1.1 if ville_match else 1.0
        else:  # NATIONAL
            return 1.0
