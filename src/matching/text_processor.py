"""
Fonctions de traitement et formatage de texte pour le matching
"""

import pandas as pd


def create_provider_text(provider_row):
    """
    Crée une représentation textuelle complète d'un prestataire
    pour générer un embedding de qualité
    
    Args:
        provider_row: pandas.Series contenant les données du prestataire
                     Colonnes attendues: Nom_Entreprise, Domaines_Expertise, 
                                       Disponibilite, Description_Service, Ville
    
    Returns:
        str: Texte formaté représentant le prestataire
    """
    parts = []
    
    # Nom de l'entreprise
    if pd.notna(provider_row.get('Nom_Entreprise')):
        parts.append(f"Entreprise: {provider_row['Nom_Entreprise']}")
    
    # Note: Ville EXCLUE de l'embedding pour éviter biais géographique
    # La géographie est gérée séparément via score_geo dans le matcher
    
    # Domaines d'expertise (très important pour le matching)
    if pd.notna(provider_row.get('Domaines_Expertise')):
        parts.append(f"Expertise: {provider_row['Domaines_Expertise']}")
    
    # Disponibilité (crucial pour l'urgence)
    if pd.notna(provider_row.get('Disponibilite')):
        parts.append(f"Disponibilité: {provider_row['Disponibilite']}")
    
    # Description du service (contient beaucoup d'informations contextuelles)
    if pd.notna(provider_row.get('Description_Service')):
        parts.append(f"Services: {provider_row['Description_Service']}")
    
    return " | ".join(parts)


def create_client_request_text(request_data):
    """
    Formate une demande client pour générer un embedding
    
    Args:
        request_data: dict ou str contenant la demande client
                     Si dict, clés possibles:
                     - type_besoin, categorie, sous_categorie
                     - service, description, message
                     - urgence, niveau_urgence
                     - ville, localisation (pour le matching géographique)
                     - details, informations_complementaires
    
    Returns:
        str: Texte formaté représentant la demande
    """
    # Si c'est déjà une chaîne, la retourner directement
    if isinstance(request_data, str):
        return request_data
    
    # Si c'est un dictionnaire, construire le texte structuré
    parts = []
    
    # Catégorie et type de besoin
    if 'categorie' in request_data:
        parts.append(f"Catégorie: {request_data['categorie']}")
    
    if 'type_besoin' in request_data:
        parts.append(f"Type: {request_data['type_besoin']}")
    
    if 'sous_categorie' in request_data:
        parts.append(f"Sous-catégorie: {request_data['sous_categorie']}")
    
    # Note: Ville et localisation EXCLUES de l'embedding
    # La géographie est gérée séparément via score_geo dans le matcher
    # Cela évite un biais géographique dans la similarité sémantique
    
    # Service demandé
    if 'service' in request_data:
        parts.append(f"Service: {request_data['service']}")
    
    # Description / Message principal
    if 'description' in request_data:
        parts.append(f"Description: {request_data['description']}")
    
    if 'message' in request_data:
        parts.append(f"Message: {request_data['message']}")
    
    # Niveau d'urgence
    if 'urgence' in request_data:
        parts.append(f"Urgence: {request_data['urgence']}")
    
    if 'niveau_urgence' in request_data:
        parts.append(f"Niveau: {request_data['niveau_urgence']}")
    
    # Détails supplémentaires
    if 'details' in request_data:
        parts.append(request_data['details'])
    
    if 'informations_complementaires' in request_data:
        parts.append(request_data['informations_complementaires'])
    
    return " | ".join(parts)


def create_need_text(need_row):
    """
    Crée une représentation textuelle d'un besoin depuis le CSV besoins.csv
    
    Args:
        need_row: pandas.Series contenant les données du besoin
                 Colonnes attendues: Categorie_Majeure, Sous_Categorie,
                                   Message_Utilisateur, Niveau_Urgence
    
    Returns:
        str: Texte formaté représentant le besoin
    """
    parts = []
    
    if pd.notna(need_row.get('Categorie_Majeure')):
        parts.append(f"Catégorie: {need_row['Categorie_Majeure']}")
    
    if pd.notna(need_row.get('Sous_Categorie')):
        parts.append(f"Type: {need_row['Sous_Categorie']}")
    
    if pd.notna(need_row.get('Niveau_Urgence')):
        parts.append(f"Urgence: {need_row['Niveau_Urgence']}")
    
    if pd.notna(need_row.get('Message_Utilisateur')):
        parts.append(f"Message: {need_row['Message_Utilisateur']}")
    
    return " | ".join(parts)


def clean_text(text):
    """
    Nettoie et normalise un texte
    
    Args:
        text: Texte à nettoyer
    
    Returns:
        str: Texte nettoyé
    """
    if not isinstance(text, str):
        return ""
    
    # Supprimer les espaces multiples
    text = " ".join(text.split())
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    return text
