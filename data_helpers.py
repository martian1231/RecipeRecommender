import pandas as pd
import ast
from utils import normalize_ingredient
import streamlit as st
import zipfile
import subprocess
import os
import shutil


def download_data(st):
    """
    Downloads files from Google Drive using gdown, moves them to the appropriate directory,
    and unzips necessary files.
    """
    # Define file IDs and destination paths
    file_ids = {
        "RAW_interactions.feather": "12K4AY4J4T2oBkVmMLLzm17FKlyQCpdeO",
        "RAW_recipes.feather": "1TPKASy5lho42ag7SRXkTKW51WGQ1z7xG",
        "bert_ingredients_embeddings.zip": "13M5FgcxAUQ2nLFmU3ja1_VkhWHFpi8x6"
    }
    
    destination_folder = "./data"
    os.makedirs(destination_folder, exist_ok=True)
    
    placeholder = st.empty()
    placeholder.info("Starting the data download process...")

    progress_bar = st.progress(0)
    
    # Download files using gdown
    for  i, data in enumerate(file_ids.items()):
        filename, file_id = data
        if not os.path.exists(f"{destination_folder}/{filename}"):
            print(f"Downloading file with ID: {file_id}")
            subprocess.run(["gdown", file_id], check=True)

            if filename == "bert_ingredients_embeddings.zip":
                # Move download file to data folder
                shutil.move(filename, destination_folder)
                
                filename = os.path.join(destination_folder, filename)
                
                # Unzip the bert_ingredients_embeddings.zip file into the data folder
                print(f"Unzipping {filename} into {destination_folder}")
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(destination_folder)
            else:
                # Move download file to data folder
                shutil.move(filename, destination_folder)
        else:
            print(f"File '{filename}' already exist")
        
        progress_bar.progress((i + 1) / len(file_ids))
    
    placeholder.empty()
    progress_bar.empty()
            
@st.cache_data
def load_data(st):
    """
    Loads recipe and user interaction data from CSV files.
    Returns two DataFrames: df_raw_recipe and df_raw_user_inter.
    """

    placeholder = st.empty()
    placeholder.info("Loading data...")

    progress_bar = st.progress(0)

    df_raw_recipe = pd.read_feather(
        "./data/RAW_recipes.feather"
    )

    progress_bar.progress(50)

    df_raw_user_inter = pd.read_feather("./data/RAW_interactions.feather")


    progress_bar.progress(75)

    df_raw_recipe = preprocess_recipes(df_raw_recipe, normalize_ingredient)

    print("Data loaded successfully")
    placeholder.empty()
    progress_bar.empty()

    return df_raw_recipe, df_raw_user_inter

def preprocess_recipes(df_raw_recipe, normalize_ingredient):
    """
    Preprocesses the raw recipe DataFrame by converting stringified lists to Python lists/sets,
    and normalizing ingredient names.
    
    Args:
        df_raw_recipe (pd.DataFrame): The raw recipe DataFrame with columns 'ingredients', 'tags', and 'steps'.
        normalize_ingredient (function): A function to normalize ingredient names.
    
    Returns:
        pd.DataFrame: The preprocessed recipe DataFrame.
    """
    # Convert stringified lists to Python objects
    df_raw_recipe['ingredients'] = df_raw_recipe['ingredients'].apply(ast.literal_eval).apply(set)
    df_raw_recipe['tags'] = df_raw_recipe['tags'].apply(ast.literal_eval).apply(set)
    df_raw_recipe['steps'] = df_raw_recipe['steps'].apply(ast.literal_eval)
    
    # Normalize ingredient names
    df_raw_recipe['ingredients'] = df_raw_recipe['ingredients'].apply(normalize_ingredient)
    
    return df_raw_recipe
