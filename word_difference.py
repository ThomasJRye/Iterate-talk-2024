import gensim.downloader as api
from scipy.spatial.distance import cosine

# Load the pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

def get_word_embedding(word: str):
    return model[word]

def find_closest_word(vector):
    return model.similar_by_vector(vector, topn=1)[0][0]

def main():
    word1 = input("Enter the first word: ")
    word2 = input("Enter the second word: ")

    if word1 not in model or word2 not in model:
        print("One of the words is not in the vocabulary.")
        return

    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)

    # Calculate the difference vector
    difference_vector = embedding1 - embedding2

    # Find the closest word to the difference vector
    closest_word = find_closest_word(difference_vector)

    print(f"The word corresponding to the difference of the embeddings is: {closest_word}")

if __name__ == "__main__":
    main()