
from ollama import chat
from ollama import ChatResponse
import pandas as pd

''' 
        SETUP
        
        terminal: ollama run deepseek-r1:1.5b
'''
MODEL = 'deepseek-r1:1.5b'
#MODEL = 'deepseek-r1:7b'
#MODEL = 'deepseek-r1:8b'
#MODEL = 'deepseek-r1:14b'
#MODEL = 'deepseek-r1:32b'
#MODEL = 'deepseek-r1:70b'
#MODEL = 'deepseek-r1:671b'

DATA_PATH = "datasets/dronology/dronology.csv"

PERSONA_MSG = [
    {"role": "system", "content": "You are a expert software engineer for Unmanned Aerial System (UAS)."},
]

MSGS = [
    {"role": "user", "content": "How do I optimize C code for an STM32?"},
    {"role": "user", "content": "Define requirements for STM32?"},
]

def main():
    
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(DATA_PATH)
    requirements = df['RequirementText']
    
    FULL_MSGS = PERSONA_MSG + MSGS
    
    response:ChatResponse = chat(
        model=MODEL, 
        options={ # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        "num_ctx": 4096,
        "seed": 0,
        "num_predict": 100,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.0,
        "repeat_last_n": 64,
        "temperature": 0.8,
        "repeat_penalty": 1.2,
        "mirostat": 0,
        "mirostat_tau": 0.5,
        "mirostat_eta": 0.1,
        },
        #stop=["."], # stop condition, e.g. "." stops generation after one sentence
        messages=FULL_MSGS
    )
    
    print(response.message.content)

if __name__ == "__main__":
    main()