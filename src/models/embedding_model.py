"""
Gestion du modèle d'embeddings pour le matching
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Classe pour gérer le modèle d'embeddings Sentence-Transformers"""
    
    def __init__(self, model_name='dangvantuan/sentence-camembert-large'):
        """
        Initialise le modèle d'embeddings
        
        Args:
            model_name: Nom du modèle Sentence-Transformers
                       Recommandés pour le français:
                       - 'dangvantuan/sentence-camembert-large' (excellent pour le français)
                       - 'paraphrase-multilingual-MiniLM-L12-v2' (rapide, multilingue)
                       - 'paraphrase-multilingual-mpnet-base-v2' (plus précis)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle en mémoire"""
        print(f"Chargement du modèle {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print(f">> Modèle {self.model_name} chargé avec succès")
    
    def encode(self, texts, show_progress=True, batch_size=32):
        """
        Génère des embeddings pour une liste de textes
        
        Args:
            texts: Liste de textes à encoder (str ou list of str)
            show_progress: Afficher une barre de progression
            batch_size: Taille des batchs pour le traitement
        
        Returns:
            numpy.ndarray: Matrice d'embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text):
        """
        Encode un seul texte (optimisé pour les requêtes individuelles)
        
        Args:
            text: Texte à encoder
        
        Returns:
            numpy.ndarray: Vecteur d'embedding (embedding_dim,)
        """
        return self.encode([text], show_progress=False)[0]
    
    def get_embedding_dimension(self):
        """Retourne la dimension des embeddings du modèle"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self):
        """Retourne les informations sur le modèle"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'max_seq_length': self.model.max_seq_length
        }
