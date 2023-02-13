import os
import openai
from dotenv import load_dotenv

#get OPENAI_API_KEY from .env file

def generate_text(prompt, max_tokens=100, temperature=0.5):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    response = openai.Completion.create(
    model="text-davinci-003",
    #prompt="Convert this text to a programmatic command:\n\nExample: Ask Constance if we need some bread\nOutput: send-msg `find constance` Do we need some bread?\n\nReach out to the ski store and figure out if I can get my skis fixed before I leave on Thursday",
    prompt=prompt,
    temperature=0,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.2,
    presence_penalty=0.0,
    stop=["input:"]
    )

    
    return(response['choices'][0]['text'])

if __name__ == "__main__":
    generate_text("This it stoey of trump ")

