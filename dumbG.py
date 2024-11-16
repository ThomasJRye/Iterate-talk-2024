import sys
import random
import os

def read_lines_from_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)

if __name__ == "__main__":
    filename = 'words.txt'
    english_words = read_lines_from_file(filename)
    generated_text = ""

    print("Press Enter to get a new word, or type 'exit' to quit.")

    while True:
        user_input = input("promt :")
        if user_input.lower() == 'exit':
            break

        generated_text += user_input + " "

        os.system('clear')  # Clear the terminal
        print("Press Enter to get a new word, or type 'exit' to quit.")

        new_word = random.choice(english_words)
        generated_text += new_word + " "
        print(generated_text, flush=True)