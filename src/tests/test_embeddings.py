from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dangvantuan/sentence-camembert-large")

sentences = [
    "C'est une personne heureuse",
    "C'est un chien heureux",
    "C'est une personne très heureuse",
    "Aujourd'hui est une journée ensoleillée"
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
print(similarities)
# [4, 4]