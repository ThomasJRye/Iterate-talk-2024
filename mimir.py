import argparse
import GPT3

parser = argparse.ArgumentParser(
    prog = 'Mimir',
    description = 'CLI for testing the Mimir conversational AI',
    epilog = 'Text at the bottom of help')
#add argument which is not required

#parser.add_argument("echo", required=False)
parser.add_argument("-v", "--verbose", help="increate output verbosity", action="store_true")
args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")

prompt = input("Hi, I am the Mimir bot and I have here to help you set goals for your life. Do you have any goals in mind?\n")
print('\n')

chatting = True
while chatting:
    output = GPT3.generate_text(prompt)
    
    prompt = input(output + "\n")
    
    if prompt == "stop":
        chatting = False
