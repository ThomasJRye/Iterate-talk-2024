import gensim.downloader as api
import numpy as np

# Load the pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

def get_word_embedding(word: str):
    return model[word]

def mean_squared_loss(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return np.mean((embedding1 - embedding2) ** 2)

def main():
    while True:
        word1 = input("Enter the first word: ")
        word2 = input("Enter the second word: ")

        if word1 not in model or word2 not in model:
            print("One of the words is not in the vocabulary.")
            return

        embedding1 = get_word_embedding(word1)
        embedding2 = get_word_embedding(word2)

        loss = mean_squared_loss(embedding1, embedding2)

        print(f"The mean squared loss between the embeddings of '{word1}' and '{word2}' is: {loss}")

if __name__ == "__main__":
    main()