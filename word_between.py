import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

def get_word_embedding(word: str):
    return model[word]

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def word_analogy(word1: str, word2: str, word3: str):
    return model.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0][0]

def visualize_embeddings(words: list):
    embeddings = np.array([get_word_embedding(word) for word in words])
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.show()

def main():
    while True:
        word1 = input("Enter the first word: ")
        word2 = input("Enter the second word: ")

        if word1 not in model or word2 not in model:
            print("One of the words is not in the vocabulary.")
            continue

        embedding1 = get_word_embedding(word1)
        embedding2 = get_word_embedding(word2)

        # Cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")

        # Word analogy
        word3 = input("Enter the third word for analogy (e.g., 'woman' for 'man is to king as woman is to ?'): ")
        if word3 not in model:
            print("The third word is not in the vocabulary.")
            continue
        analogy_word = word_analogy(word1, word2, word3)
        print(f"'{word1}' is to '{word2}' as '{word3}' is to '{analogy_word}'")

        # Visualization
        words_to_visualize = [word1, word2, word3, analogy_word]
        visualize_embeddings(words_to_visualize)

if __name__ == "__main__":
    main()