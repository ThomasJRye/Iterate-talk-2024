import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from spellchecker import SpellChecker

# Load the pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

def get_word_embedding(word: str):
    return model[word]

def cosine_similarity(word1: str, word2: str) -> float:
    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def mean_squared_loss(word1: str, word2: str) -> float:
    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)
    return np.mean((embedding2 - embedding1) ** 2)

def most_similar(word: str, topn=5):
    return model.most_similar(word, topn=topn)

def word_analogy(word1: str, word2: str, word3: str):
    return model.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0][0]

def is_correctly_spelled(word):
    spell = SpellChecker()
    return word in spell

def visualize_embeddings(words: list):
    embeddings = np.array([get_word_embedding(word) for word in words])
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.show()

def find_closest_word(model, target_embedding):
    closest_word = model.most_similar([target_embedding], topn=1)[0][0]
    return closest_word

def has_no_uppercase(s):
    return not any(char.isupper() for char in s)

def most_different_word(word: str) -> str:
    target_embedding = get_word_embedding(word)
    most_different_words = model.most_similar(negative=[target_embedding], topn=5000)

    for similar_word, similarity in most_different_words:
        if '_' not in similar_word:
            if '.' not in similar_word:
                if has_no_uppercase(similar_word):
                    if word.isalpha():
                        if is_correctly_spelled(similar_word):
                            return similar_word +  " similarity: " + similarity.__str__()
        
    return "No word found"

def main():
    while True:
        word1 = input("Enter the first word: ")
        if word1 == 'exit()':
            break

        word2 = input("Enter the second word: ")
        if word2 == 'exit()':
            break

        if word1 not in model or word2 not in model:
            print("One of the words is not in the vocabulary.")
            continue

        # Calculate and print mean squared loss
        loss = mean_squared_loss(word1, word2)
        print(f"Mean squared loss between '{word1}' and '{word2}': {loss}")

        # Cosine similarity
        similarity = cosine_similarity(word1, word2)
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