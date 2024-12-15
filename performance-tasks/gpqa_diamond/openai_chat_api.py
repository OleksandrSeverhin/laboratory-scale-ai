import openai
from typing import Iterable
from os import getenv

# Set the OpenAI API key
openai.api_key = getenv('OPENAI_API_KEY')

class ChatBot:
    """
    A class for interacting with the OpenAI Chat API.
    """

    def __init__(self,
                 model: str='gpt-3.5-turbo',
                 system_prompt: str='You are a helpful assistant.') -> None:
        """
        Initialize a ChatBot object, setting system prompt if preferred.
        """

        self.model = model
        self.system_prompt = [{'role': 'system', 'content': system_prompt}]

    def generate(self,
                 messages: Iterable,
                 new_api: bool=True) -> str:
        """
        Query the OpenAI Chat API to generate a response to the user's input.
        """

        if new_api:
            completion = openai.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            output = completion.choices[0].message.content

        else:
            # Generate the bot's response
            output = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
            )['choices'][0]['message']['content']

        return output

class DialogueBot(ChatBot):

    def __init__(self,
                 model: str='gpt-3.5-turbo',
                 system_prompt: str='You are a helpful assistant.',
                 history: Iterable=None) -> None:
        """
        Initialize a DialogueBot object, setting system prompt if preferred.
        """

        super().__init__(model, system_prompt)
        self.history = history if history is not None else []

    def respond_to_user(self,
                        input: str) -> tuple:
        """
        Respond to the user's input, while logging the conversation history for possible display in a UI.
        This is better for keeping track of a running conversation in a UI.
        """

        # Add the user input to the history
        self.history.append({'role': 'user', 'content': input})
        messages = self.system_prompt + self.history

        # Generate the bot's response
        output = self.generate(messages)

        # Add the bot's response to the history - by default, it is added to the history
        self.history.append({'role': 'assistant', 'content': output})
        response = [(self.history[i]['content'], self.history[i+1]['content']) for i in range(0, len(self.history)-1, 2)]

        # Return the response and the history
        return response, self.history

    def return_bot_response(self,
                        input: str,
                        log_history: bool=False) -> tuple:
        """
        Return the bot's response to the user's input; by default, does not add anything to the conversation history.
        This is useful for generating responses to tasks that do not require a conversation history.
        """

        # Add the user input to the model prompt
        messages = self.system_prompt + self.history + [{'role': 'user', 'content': input}]

        # Generate the bot's response
        output = self.generate(messages)

        # Add the bot's response to the history - by default, this is not added to the history
        if log_history:
            self.history.append({'role': 'user', 'content': input})
            self.history.append({'role': 'assistant', 'content': output})

        # Return the bot's response
        return output

    def change_system_prompt(self,
                             system_prompt: str) -> None:
        """
        Change the system-level prompt governing bot behavior at a high level.
        """

        self.system_prompt = [{'role': 'system', 'content': system_prompt}]