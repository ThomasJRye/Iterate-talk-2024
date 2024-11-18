from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Save the model
model.load("bible_word2vec.model")

# Example: Get the vector for a specific word
word_vector = model.wv['god']
print("Word vector for 'god':", word_vector.size)

# Example: Find most similar words
similar_words = model.wv.most_similar('god', topn=5)
print("Most similar words to 'god':", similar_words)

print("number of words: ", len(model.wv.index_to_key))

print("vocab: ", model.wv.index_to_key)