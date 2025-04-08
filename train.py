import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import nltk
from nltk.chat.util import Chat, reflections
from tqdm import tqdm
import time

# Make sure to download necessary resources for nltk
nltk.download('punkt')

# Path to your training data file (text file with examples)
train_file = "motivational_examples.txt"

# Define the training function
def fine_tune_gpt2(train_file, output_dir="./fine-tuned-gpt2"):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train model
    tqdm(trainer.train())
    
    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

# Train the model
model_dir = fine_tune_gpt2(train_file)
print(f"Model saved to {model_dir}")