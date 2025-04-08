import random
import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.chat.util import Chat, reflections

# Make sure to download necessary resources for nltk
nltk.download('punkt')

# Motivational Responses
motivational_quotes = [
    "You are capable of amazing things!",
    "Believe in yourself and all that you are.",
    "Success is the sum of small efforts, repeated day in and day out.",
    "Hardships often prepare ordinary people for an extraordinary destiny.",
    "You are never too old to set another goal or to dream a new dream.",
    "The only way to do great work is to love what you do.",
    "Believe you can and you're halfway there.",
    "You are stronger than you think."
]

# Define the chatbot's behavior with GPT-2 integration
class MotivationalBot:
    def __init__(self):
        self.motivational_phrases = motivational_quotes

        # Load pre-trained GPT-2 model and tokenizer from Hugging Face
        print("sdfghjkjhgfefhjkhg")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("sdfgtyeds")

        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        print("sadsggeg")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_gpt_response(self, user_input):
        inputs = self.tokenizer(user_input + self.tokenizer.eos_token, return_tensors='pt', padding=True, truncation=True)

        # Create the attention mask manually since pad_token == eos_token
        attention_mask = (inputs['input_ids'] != self.tokenizer.pad_token_id).long()

        # Generate the response using GPT-2 with the attention mask
        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=attention_mask,  # Use the attention mask
            max_length=100, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and return the generated response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def get_response(self, user_input):
        # Check if the input contains keywords indicating a motivational response is needed
        if any(word in user_input.lower() for word in ["sad", "down", "bad", "depressed", "tired"]):
            return random.choice(self.motivational_phrases)
        else:
            # Use GPT-2 to generate a more dynamic response
            return self.generate_gpt_response(user_input)

# Function to interact with the bot

def chat():
    print("Hello! I am your motivational bot. How can I help you today?")
    print("Type 'quit' to exit the conversation.")
    bot = MotivationalBot()
    
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye! Keep going strong!")
            break
        else:
            response = bot.get_response(user_input)
            print("Bot: " + response)

chat()
