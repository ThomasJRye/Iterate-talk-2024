from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# Load and preprocess the text
def preprocess_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Read and convert to lowercase
    
    # Tokenize and remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # Keep only words (remove numbers/punctuation)
    
    return tokens

# Split tokens into sentences for Word2Vec
def split_into_sentences(tokens, sentence_length=100):
    return [tokens[i:i + sentence_length] for i in range(0, len(tokens), sentence_length)]

# Load and preprocess the Bible text
tokens = preprocess_text('bible.txt')
sentences = split_into_sentences(tokens)

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=3, workers=4)

# Save the model
model.save("bible_word2vec.model")

# Example: Get the vector for a specific word
word_vector = model.wv['god']
print("Word vector for 'god':", word_vector.size)

# Example: Find most similar words
similar_words = model.wv.most_similar('god', topn=5)
print("Most similar words to 'god':", similar_words)

print("number of words: ", len(model.wv.index_to_key))

print("vocab: ", model.wv.index_to_key)