"""
Test du nouveau système de scoring géographique
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.ner_extractor import NERExtractor
from math import exp


def test_geo_score():
    """
    Teste le nouveau score géographique avec différentes configurations
    """
    ner = NERExtractor()
    
    print("="*80)
    print("TEST DU SCORE GÉOGRAPHIQUE V2.0")
    print("Formule: score_geo = exp(-alpha * distance)")
    print("="*80)
    
    # Scénarios de test
    scenarios = [
        {
            'nom': 'GARDE D\'ENFANTS (impact_geo=2, proximité critique)',
            'ville_besoin': 'Paris',
            'tests': [
                ('Paris', 0),      # Même ville
                ('Versailles', 14),  # Très proche (14 km)
                ('Lyon', 391),     # Loin
                ('Marseille', 660) # Très loin
            ],
            'impact_geo': 2,
            'alpha': 0.05
        },
        {
            'nom': 'LOGEMENT (impact_geo=1, proximité utile)',
            'ville_besoin': 'Marseille',
            'tests': [
                ('Marseille', 0),
                ('Toulon', 49),
                ('Nice', 159),
                ('Paris', 660)
            ],
            'impact_geo': 1,
            'alpha': 0.015
        },
        {
            'nom': 'PRÊT BANCAIRE (impact_geo=0, service en ligne)',
            'ville_besoin': 'Lille',
            'tests': [
                ('Lille', 0),
                ('Paris', 204),
                ('Marseille', 831),
                ('Nice', 801)
            ],
            'impact_geo': 0,
            'alpha': 0.0
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"SCÉNARIO: {scenario['nom']}")
        print(f"Besoin ville: {scenario['ville_besoin']}")
        print(f"impact_geo = {scenario['impact_geo']} (alpha = {scenario['alpha']})")
        print(f"{'='*80}\n")
        
        print(f"{'Prestataire Ville':<20} {'Distance':<12} {'Score Géo':<12} {'Interprétation'}")
        print(f"{'-'*80}")
        
        for ville_presta, distance in scenario['tests']:
            score = ner.calculate_geo_score(
                scenario['ville_besoin'],
                ville_presta,
                scenario['impact_geo']
            )
            
            # Interprétation
            if score >= 0.9:
                interp = "✅ Excellent"
            elif score >= 0.7:
                interp = "✓ Bon"
            elif score >= 0.5:
                interp = "~ Moyen"
            elif score >= 0.3:
                interp = "⚠ Faible"
            else:
                interp = "❌ Très faible"
            
            # Calcul théorique
            if scenario['alpha'] == 0:
                score_theo = 1.0
            else:
                score_theo = exp(-scenario['alpha'] * distance)
            
            print(f"{ville_presta:<20} {distance:>4.0f} km      {score:>6.4f} ({score*100:>5.1f}%)  {interp}")
    
    # Test des cas particuliers
    print(f"\n{'='*80}")
    print("CAS PARTICULIERS")
    print(f"{'='*80}\n")
    
    print("1. Besoin SANS ville détectée (impact_geo=1):")
    score = ner.calculate_geo_score(None, 'Paris', 1)
    print(f"   score_geo = {score:.4f} (pénalité légère pour incertitude)\n")
    
    print("2. Service en ligne (impact_geo=0):")
    score = ner.calculate_geo_score('Paris', 'Marseille', 0)
    print(f"   score_geo = {score:.4f} (distance non pertinente)\n")
    
    print("3. Même ville (impact_geo=2):")
    score = ner.calculate_geo_score('Lyon', 'Lyon', 2)
    print(f"   score_geo = {score:.4f} (score maximal)\n")
    
    # Calcul de distances caractéristiques
    print(f"\n{'='*80}")
    print("DISTANCES CARACTÉRISTIQUES PAR IMPACT_GEO")
    print(f"{'='*80}\n")
    
    from math import log
    
    impacts = [
        (0, 0.0, "Services en ligne"),
        (1, 0.015, "Proximité utile"),
        (2, 0.05, "Proximité critique")
    ]
    
    print(f"{'Impact':<8} {'Alpha':<8} {'Type':<20} {'Dist 50%':<12} {'Dist 10%':<12} {'Dist 1%'}")
    print(f"{'-'*80}")
    
    for impact, alpha, type_service in impacts:
        if alpha == 0:
            d50 = "∞"
            d10 = "∞"
            d1 = "∞"
        else:
            d50 = -log(0.5) / alpha
            d10 = -log(0.1) / alpha
            d1 = -log(0.01) / alpha
            d50 = f"{d50:.0f} km"
            d10 = f"{d10:.0f} km"
            d1 = f"{d1:.0f} km"
        
        print(f"{impact:<8} {alpha:<8.3f} {type_service:<20} {d50:<12} {d10:<12} {d1}")
    
    print("\n✅ Tests terminés")


if __name__ == "__main__":
    test_geo_score()
