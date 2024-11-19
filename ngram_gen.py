import sys
import random
import os
from collections import defaultdict, Counter
import nltk
from nltk import trigrams, word_tokenize
import argparse
import re

# Download necessary NLTK resources
nltk.download('punkt')


def remove_chapter_verse_markers(text):
    # Define the regex pattern to match chapter and verse markers
    pattern = r'\b\d+:\d+\b'
    # Use re.sub to replace the matched patterns with an empty string
    cleaned_text = re.sub(pattern, '', text)
    # Remove any extra spaces left after removing the markers
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Example usage
if __name__ == "__main__":
    with open('bible.txt', 'r') as file:
        text = file.read()
    
    cleaned_text = remove_chapter_verse_markers(text)
    
    with open('cleaned_bible.txt', 'w') as file:
        file.write(cleaned_text)
    
    print("Chapter and verse markers removed and saved to 'cleaned_bible.txt'")
class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)

    def preprocess(self, text):
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def train(self, text):
        text = self.preprocess(text)
        tokens = text.split()
        for i in range(len(tokens) - self.n):
            ngram = tuple(tokens[i:i+self.n])
            next_token = tokens[i+self.n]
            self.ngrams[ngram][next_token] += 1

    def generate(self, seed, length=50):
        seed = self.preprocess(seed)
        current = tuple(seed.split()[:self.n])
        result = list(current)
        for _ in range(length):
            next_token = self._predict_next(current)

            if next_token is None:
                break
            result.append(next_token)
            current = tuple(result[-self.n:])
        return ' '.join(result)

    def _predict_next(self, ngram):
        next_tokens = self.ngrams.get(ngram)
        if not next_tokens:
            return None
        total = sum(next_tokens.values())
        rand = random.randint(1, total)
        for token, count in next_tokens.items():
            rand -= count
            if rand <= 0:
                return token

def read_lines_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()  # Read the whole content to build the model
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)

if __name__ == "__main__":
    filename='training.txt'
    
    input_text = remove_chapter_verse_markers(read_lines_from_file(filename))

    model = NgramModel(n=2)  # Use a smaller N-gram size
    model.train(input_text)

    context = ""
    while True:
        user_input = input("promt: ")
        if user_input.lower() == 'exit':
            break

        context += user_input + " "

        os.system('clear')  # Clear the terminal
        print("Press Enter to get a new word, or type 'exit' to quit.")

        new_word = model.generate(context)
        context += new_word + " "
        print(context, flush=True)