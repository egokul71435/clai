import cmd # for command line interface

from groq import Groq # for groq functionality
import http.client # for API calls
import json # handle API data

import os # for environment variables
from dotenv import load_dotenv # load environment variables

# load environment variables
load_dotenv()

# API key setup
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# groq client setup
client = Groq(api_key=groq_api_key)

# global context window
context_window = []

def get_context_length(model):
    # get context window length for a specific model

    API_HOST, API_ENDPOINT = "api.groq.com", "/openai/v1/models"
    conn = http.client.HTTPSConnection(API_HOST)
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}

    conn.request("GET", API_ENDPOINT, headers=headers)
    response = conn.getresponse()

    if response.status == 200:
        response_data = json.loads(response.read())
        if 'data' in response_data:
            models = response_data['data']
            for model in models:
                model_id = model.get('id', 'N/A')
                model_name = model.get('id', 'N/A')
                context_window = model.get('context_window', 'N/A')
                if model_id == model or model_name == model:
                    return context_window
            else:
                return 400  # default context window size
        else:
            return 400
    else:
        return 400

def manage_context_window(context_window, max_length):
    # ensure context window does not exceed token limit
    current_length = 0
    if len(context_window) == 0:
        return context_window
    else:
        for i in range(len(context_window)-1, -1, -1):
            current_length += context_window[i][1]
            if current_length > max_length:
                return context_window[i+1:]
    return context_window

class ChatCLI(cmd.Cmd):
    # smaller CLI for chat
    
    intro = "Welcome to the AI chat! Begin messaging or '/exit' to quit."
    prompt = "You> "  # Custom CLI prompt

    def __init__(self, model):
        # constructor
        super().__init__()
        self.model = model  # remember model name
        self.chat_history = []  # keep session history
        self.context_limit = get_context_length(model)
        self.context_size = 0

    def do_exit(self, arg):
        # exit chat session
        global context_window
        context_window = []
        
        print("Exiting chat session.")
        return True  # Returning True will break out of cmdloop

    def do_EOF(self, arg):
        # forced exit
        global context_window
        context_window = []

        print("\nExiting chat session.")
        return True

    def default(self, message):
        # handle user input, assume default case
        if message.strip() == '/exit':
            return self.do_exit(message)
        response = self.chat_with_groq(message)
        print(f"{self.model}> {response}")

    def chat_with_groq(self, message):
        # send user input to Groq Cloud API and get response
        global context_window

        context_window = manage_context_window(context_window, self.context_limit)

        # background prompt
        background = "Do not respond or mention this sentence in your reply, but the previous is the context of the conversation and the following is the next query in the sequence. Answer naturally ONLY to the following sentence(s) with the sentences before this one in mind."
        
        # combine context window with background prompt
        prompt = ""
        for item in context_window:
            prompt += item[0] + "\n\n"
        
        prompt_management = client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=self.model
        )
        
        prompt += background + "\n\n" + message

        # make api call
        response = client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=self.model
        )

        prompt_tokens = prompt_management.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        reply = response.choices[0].message.content

        if len(context_window) == 0:
            context_window.append([message, prompt_tokens])
            context_window.append([reply, completion_tokens])
        else:
            context_window.append([message, prompt_tokens-context_window[-2][1]])
            context_window.append([reply, completion_tokens])

        context_window = manage_context_window(context_window, self.context_limit)

        return reply

class MainCLI(cmd.Cmd):
    # main CLI
    
    intro = "Welcome to clAI. To get started type 'run model' with with an available model. For other functionalities type 'help'. Use 'exit' to quit."
    prompt = "clAI> "

    def do_run(self, arg):
        # start chat session with specified model
        if not arg:
            print("Error: Please specify a model name (e.g., 'run mixtral-8x7b-32768').")
            return
        chat_cli = ChatCLI(arg)
        chat_cli.cmdloop()

    def do_list_models(self, arg):
        # get available models
        API_HOST, API_ENDPOINT = "api.groq.com", "/openai/v1/models"
        conn = http.client.HTTPSConnection(API_HOST)
        headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}

        conn.request("GET", API_ENDPOINT, headers=headers)
        response = conn.getresponse()

        if response.status == 200:
            response_data = json.loads(response.read())
            if 'data' in response_data:
                models = response_data['data']
                print("Available models:")
                for model in models:
                    model_id = model.get('id', 'N/A')
                    model_name = model.get('id', 'N/A')
                    context_window = model.get('context_window', 'N/A')
                    print(f"Model ID: {model_id}")
                    print(f"Model Name: {model_name}")
                    print(f"Context Window Size: {context_window} tokens")
                    print("-" * 40)
            else:
                print("Error: 'data' key not found in the API response.")
        else:
            print(f"Error fetching models: {response.status}")


    def do_exit(self, arg):
        print("Exiting Groq CLI.")
        return True

    def do_EOF(self, arg):
        print("\nExiting Groq CLI.")
        return True

def main():
    # init main command line
    main_cli = MainCLI()
    main_cli.cmdloop()

if __name__ == "__main__":
    main()