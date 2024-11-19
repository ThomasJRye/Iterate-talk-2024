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
    parser = argparse.ArgumentParser(description="Generate text using an N-gram model.")
    parser.add_argument("seed_text", type=str, help="Seed text to start the generation")
    args = parser.parse_args()

    input_text = read_lines_from_file(filename)
    seed_text = args.seed_text

    model = NgramModel(n=2)  # Use a smaller N-gram size
    model.train(input_text)
    generated_text = model.generate(seed_text)

    print(f"Generated text: {generated_text}")