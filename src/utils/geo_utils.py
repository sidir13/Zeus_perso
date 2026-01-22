"""
Utilitaire pour calculer les distances géographiques entre villes françaises
Utilise geopy pour un géocodage automatique et des distances précises
"""

from typing import Dict, Optional, Tuple
from math import radians, cos, sin, asin, sqrt

# Essayer d'importer geopy (plus précis et automatique)
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("⚠️  geopy non installé. Installation recommandée : pip install geopy")


# Cache pour éviter de géocoder plusieurs fois la même ville
_geocode_cache: Dict[str, Optional[Tuple[float, float]]] = {}

# Géocodeur Nominatim (OpenStreetMap)
if GEOPY_AVAILABLE:
    _geolocator = Nominatim(user_agent="zeus_armee_matching", timeout=3)


# Coordonnées GPS (latitude, longitude) des principales villes françaises
COORDONNEES_VILLES = {
    'Paris': (48.8566, 2.3522),
    'Lyon': (45.7640, 4.8357),
    'Marseille': (43.2965, 5.3698),
    'Toulouse': (43.6047, 1.4442),
    'Lille': (50.6292, 3.0573),
    'Bordeaux': (44.8378, -0.5792),
    'Nice': (43.7102, 7.2620),
    'Nantes': (47.2184, -1.5536),
    'Strasbourg': (48.5734, 7.7521),
    'Montpellier': (43.6108, 3.8767),
    'Rennes': (48.1173, -1.6778),
    'Toulon': (43.1242, 5.9280),
    'Grenoble': (45.1885, 5.7245),
    'Dijon': (47.3220, 5.0415),
    'Angers': (47.4784, -0.5632),
    'Brest': (48.3905, -4.4860),
    'Le Mans': (48.0077, 0.1984),
    'Metz': (49.1193, 6.1757),
    'Reims': (49.2583, 4.0317),
    'Orléans': (47.9029, 1.9093),
    'Bourges': (47.0816, 2.3987),
    'Vendée': (46.6706, -1.4269),  # Approximation: La Roche-sur-Yon
    'Versailles': (48.8049, 2.1204),
    'Rouen': (49.4432, 1.0993),
    'Mulhouse': (47.7508, 7.3359),
    'Caen': (49.1829, -0.3707),
    'Nancy': (48.6921, 6.1844),
    'Saint-Étienne': (45.4397, 4.3872),
    'Avignon': (43.9493, 4.8055),
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule la distance en km entre deux points GPS avec la formule de Haversine
    
    Args:
        lat1, lon1: Latitude et longitude du point 1
        lat2, lon2: Latitude et longitude du point 2
        
    Returns:
        Distance en kilomètres
    """
    # Rayon de la Terre en km
    R = 6371.0
    
    # Conversion en radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Différences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Formule de Haversine
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return R * c


def get_distance_entre_villes(ville1: str, ville2: str) -> Optional[float]:
    """
    Retourne la distance en km entre deux villes françaises
    
    Stratégie :
    1. Essayer avec geopy (géocodage automatique, toutes villes)
    2. Fallback sur base hardcodée si geopy échoue
    
    Args:
        ville1: Nom de la première ville
        ville2: Nom de la deuxième ville
        
    Returns:
        Distance en km, ou None si impossible de calculer
    """
    # Méthode 1 : Utiliser geopy (préféré)
    if GEOPY_AVAILABLE:
        try:
            coords1 = _geocode_ville(ville1)
            coords2 = _geocode_ville(ville2)
            
            if coords1 and coords2:
                # Distance geodesic (ellipsoïde WGS-84, plus précis que Haversine)
                distance = geodesic(coords1, coords2).kilometers
                return distance
        except Exception as e:
            # Si geopy échoue, fallback sur méthode classique
            pass
    
    # Méthode 2 : Fallback sur base hardcodée
    ville1_clean = ville1.strip()
    ville2_clean = ville2.strip()
    
    if ville1_clean not in COORDONNEES_VILLES or ville2_clean not in COORDONNEES_VILLES:
        return None
    
    lat1, lon1 = COORDONNEES_VILLES[ville1_clean]
    lat2, lon2 = COORDONNEES_VILLES[ville2_clean]
    
    return haversine_distance(lat1, lon1, lat2, lon2)


def _geocode_ville(ville: str) -> Optional[Tuple[float, float]]:
    """
    Géocode une ville française pour obtenir ses coordonnées GPS
    
    Utilise un cache pour éviter de géocoder plusieurs fois la même ville
    
    Args:
        ville: Nom de la ville
        
    Returns:
        Tuple (latitude, longitude) ou None si introuvable
    """
    if not GEOPY_AVAILABLE:
        return None
    
    ville_clean = ville.strip()
    
    # Vérifier le cache
    if ville_clean in _geocode_cache:
        return _geocode_cache[ville_clean]
    
    try:
        # Géocoder avec contexte français
        location = _geolocator.geocode(f"{ville_clean}, France", language='fr')
        
        if location:
            coords = (location.latitude, location.longitude)
            _geocode_cache[ville_clean] = coords
            return coords
        else:
            _geocode_cache[ville_clean] = None
            return None
            
    except Exception as e:
        _geocode_cache[ville_clean] = None
        return None


def get_liste_villes() -> list:
    """
    Retourne la liste de toutes les villes disponibles dans la base hardcodée
    Note: Avec geopy, n'importe quelle ville française peut être utilisée
    """
    return list(COORDONNEES_VILLES.keys())


def is_geopy_available() -> bool:
    """
    Vérifie si geopy est disponible
    """
    return GEOPY_AVAILABLE


if __name__ == "__main__":
    # Tests
    print("=" * 80)
    print("🗺️  TEST DU SYSTÈME DE CALCUL DE DISTANCES")
    print("=" * 80)
    
    # Afficher le statut de geopy
    if GEOPY_AVAILABLE:
        print("\n✅ geopy installé : Géocodage automatique activé")
        print("   → Toutes les villes françaises supportées")
    else:
        print("\n⚠️  geopy non installé : Utilisation base hardcodée uniquement")
        print("   → 29 villes supportées")
        print("   → Pour installer : pip install geopy")
    
    print(f"\n{'='*80}")
    print("TESTS DE DISTANCES")
    print(f"{'='*80}\n")
    
    test_cases = [
        ('Paris', 'Lyon'),
        ('Marseille', 'Nice'),
        ('Lille', 'Bordeaux'),
        ('Paris', 'Nancy'),
        ('Toulon', 'Marseille'),
    ]
    
    # Tester avec villes hardcodées
    print("📍 Villes de la base hardcodée :")
    for ville1, ville2 in test_cases:
        distance = get_distance_entre_villes(ville1, ville2)
        if distance:
            print(f"   {ville1:15s} → {ville2:15s} : {distance:6.1f} km")
        else:
            print(f"   {ville1:15s} → {ville2:15s} : ❌ Non trouvé")
    
    # Tester avec villes non hardcodées (si geopy disponible)
    if GEOPY_AVAILABLE:
        print("\n📍 Nouvelles villes (via géocodage geopy) :")
        new_cities = [
            ('Montpellier', 'Perpignan'),
            ('Clermont-Ferrand', 'Limoges'),
            ('Angers', 'Le Mans'),
        ]
        
        for ville1, ville2 in new_cities:
            distance = get_distance_entre_villes(ville1, ville2)
            if distance:
                print(f"   {ville1:15s} → {ville2:15s} : {distance:6.1f} km")
            else:
                print(f"   {ville1:15s} → {ville2:15s} : ❌ Non trouvé")
    
    print(f"\n{'='*80}")
