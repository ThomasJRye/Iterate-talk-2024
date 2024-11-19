import sys
import random
import os
from collections import defaultdict
import nltk
from nltk import trigrams, word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')

def build_trigram_model(corpus):
    # Tokenize the text
    words = nltk.word_tokenize(corpus)
    # Create trigrams
    tri_grams = list(trigrams(words))
    # Build a trigram model
    model = defaultdict(lambda: defaultdict(lambda: 0))
    # Count frequency of co-occurrence
    for w1, w2, w3 in tri_grams:
        model[(w1, w2)][w3] += 1
    # Transform the counts into probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
    return model

def predict_next_word(model, w1, w2):
    next_word_candidates = model.get((w1, w2))
    if next_word_candidates:
        return max(next_word_candidates, key=next_word_candidates.get)
    else:
        return None

def read_lines_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()  # Read the whole content to build the model
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)

if __name__ == "__main__":
    filename = 'words.txt'
    corpus = read_lines_from_file(filename)
    trigram_model = build_trigram_model(corpus)
    generated_text = ""

    print("Press Enter to get a new word, or type 'exit' to quit.")

    while True:
        user_input = input("prompt: ")
        if user_input.lower() == 'exit':
            break

        user_input_tokens = user_input.split()
        if len(user_input_tokens) >= 2:
            w1, w2 = user_input_tokens[-2], user_input_tokens[-1]
            new_word = predict_next_word(trigram_model, w1, w2)
        else:
            new_word = None

        if new_word:
            generated_text += new_word + " "
        else:
            new_word = random.choice(corpus.split())  # Fall back to a random word if no prediction is available
            generated_text += new_word + " "

        os.system('clear')  # Clear the terminal
        print("Press Enter to get a new word, or type 'exit' to quit.")
        print(generated_text, flush=True)