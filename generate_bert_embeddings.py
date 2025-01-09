import pandas as pd
import ast
from transformers import BertTokenizer, BertModel
import torch
import json
import os

# Load the dataset
df = pd.read_csv('./data/RAW_recipes.csv')

# Destination folder
output_folder = "./data/bert_ingredients_embeddings"
os.makedirs(output_folder, exist_ok=True)


# Parse the 'ingredients' column from strings to sets
df['ingredients'] = df.ingredients.apply(lambda x: set(ast.literal_eval(x)))

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Iterate through each recipe's ID and ingredients
for id, data in df[["id", "ingredients"]].iterrows():
    recipe_id, ingredients = data  # Extract recipe ID and ingredients

    # Tokenize and encode each ingredient into BERT-compatible format
    tokenized = tokenizer(
        list(ingredients), 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    
    # Pass the tokenized data through the BERT model to get embeddings
    outputs = model(**tokenized)
    
    # Extract per-token embeddings from the last hidden layer of the model
    token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, tokens, 768)
    
    # Perform mean pooling across tokens to get a single vector for each ingredient
    mean_embedding = torch.mean(token_embeddings, dim=1)  # Shape: (batch_size, 768)
    
    # Aggregate embeddings of all ingredients into a single embedding for the recipe
    sentence_embedding = torch.mean(mean_embedding, dim=0)  # Final embedding (1, 768)
    
    # Convert the embedding to a dictionary for saving
    data_dict = {"embedding": sentence_embedding.tolist()}
    
    # Save the embedding as a JSON file for the recipe
    with open(f'{output_folder}/{recipe_id}.json', 'w') as f:
        json.dump(data_dict, f)
