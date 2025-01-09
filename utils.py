import re
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Initialize BERT components
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Initialize and download necessary NLTK data
@st.cache_data
def load_nltk():
    nltk.download('wordnet')

load_nltk()
lemmatizer = WordNetLemmatizer()

def normalize_ingredient(ingredient_set):
    """
    Normalizes a set of ingredients by converting to lowercase, 
    removing non-alphabetic characters, and lemmatizing.
    """
    ingredient_set_cleaned = set()
    for ingredient in ingredient_set:
        ingredient = ingredient.lower()  # Convert to lowercase
        ingredient = re.sub(r'[^a-z\s]', '', ingredient)  # Remove punctuation
        ingredient_set_cleaned.add(ingredient)
    return set([lemmatizer.lemmatize(word) for word in ingredient_set_cleaned])



def get_embedding(user_ingredients):
    """Generate BERT embeddings for user-provided ingredients."""
    tokenized = tokenizer(list(user_ingredients), padding=True, truncation=True, return_tensors='pt')
    outputs = model(**tokenized)
    token_embeddings = outputs.last_hidden_state
    mean_embeddings = torch.mean(token_embeddings, dim=1)
    sentence_embedding = torch.mean(mean_embeddings, dim=0)
    return sentence_embedding.tolist()



def filter_recipes_by_time(df_raw_recipe, time_limit_pref=None):
    """
    Filters recipes based on a given time limit.

    Parameters:
        df_raw_recipe (pd.DataFrame): The raw recipe dataset.
        time_limit_pref (int, optional): Maximum time (in minutes) for preparing a recipe. 
                                         If None, no filtering is applied.

    Returns:
        pd.DataFrame: Filtered recipe dataset.
    """
    if time_limit_pref:
        df_filtered = df_raw_recipe[df_raw_recipe['minutes'] <= time_limit_pref]
    else:
        df_filtered = df_raw_recipe

    if not len(df_filtered):
        print(f"No recipes found that can be prepared in {time_limit_pref} minutes.")
    
    return df_filtered


def filter_recipes_by_diet(df_filter_time, diet_pref=None, diet_pref_len=0):
    """
    Filters recipes based on a user's diet preferences.

    Parameters:
        df_filter_time (pd.DataFrame): The dataset filtered by time.
        diet_pref (set, optional): A set of diet tags preferred by the user. 
                                   If None, no diet-based filtering is applied.
        diet_pref_len (int, optional): Minimum number of matching diet tags required.

    Returns:
        pd.DataFrame: Filtered recipe dataset based on diet preferences.
    """
    if diet_pref:
        # Calculate the overlap of diet tags for each recipe
        df_filter_time["diet_overlap"] = df_filter_time["tags"].apply(
            lambda tag: len(set(tag) & diet_pref)
        )
        # Filter recipes based on the overlap threshold
        df_filter_diet = df_filter_time[df_filter_time["diet_overlap"] >= diet_pref_len]
    else:
        df_filter_diet = df_filter_time

    return df_filter_diet


def filter_recipes_by_cuisine(df_filter_diet, cuisine_pref=None, cuisine_pref_len=0):
    """
    Filters recipes based on a user's cuisine preferences.

    Parameters:
        df_filter_diet (pd.DataFrame): The dataset filtered by diet preferences.
        cuisine_pref (set, optional): A set of preferred cuisines. If None, no cuisine-based filtering is applied.
        cuisine_pref_len (int, optional): Minimum number of matching cuisine tags required.

    Returns:
        pd.DataFrame: Filtered recipe dataset based on cuisine preferences.
    """
    if cuisine_pref:
        # Calculate the overlap of cuisine tags for each recipe
        df_filter_diet["cuisine_overlap"] = df_filter_diet["tags"].apply(
            lambda cuisines: len(set(cuisines) & cuisine_pref)
        )
        # Filter recipes based on the overlap threshold
        df_filter_cuisine = df_filter_diet[df_filter_diet["cuisine_overlap"] >= cuisine_pref_len]
    else:
        df_filter_cuisine = df_filter_diet

    return df_filter_cuisine


def filter_recipes_by_ingredients(df_filter_cuisine, user_ingredients=None, ingredients_percentage_match=0.5, st= None):
    """
    Filters recipes based on user-provided ingredients and overlap percentage.

    Parameters:
        df_filter_cuisine (pd.DataFrame): The dataset filtered by cuisine preferences.
        user_ingredients (set, optional): A set of ingredients provided by the user. 
                                          If None, no ingredient-based filtering is applied.
        ingredients_percentage_match (float, optional): Minimum overlap percentage required 
                                                        between recipe ingredients and user-provided ingredients.

    Returns:
        pd.DataFrame: Filtered recipe dataset based on ingredient overlap.
    """
    df_filter_ingredient = None

    if user_ingredients:
        # Calculate ingredient overlap
        df_filter_cuisine["ingredient_overlap"] = df_filter_cuisine["ingredients"].apply(
            lambda ingredients: len(ingredients & user_ingredients) / len(user_ingredients)
        )
        # Filter recipes based on overlap percentage
        df_filter_ingredient = df_filter_cuisine[
            df_filter_cuisine["ingredient_overlap"] >= ingredients_percentage_match
        ]

        # Sort by overlap and select top 15 recipes
        df_filter_ingredient = df_filter_ingredient.sort_values(
            by="ingredient_overlap", ascending=False
        ).head(15)

    # Handle cases with no valid recipes or no user-provided ingredients
    if df_filter_ingredient is None or df_filter_ingredient.empty:
        if user_ingredients:
            print(f"No recipes found with an ingredient overlap percentage of {ingredients_percentage_match}.")
            st.toast(f"No recipes found with an ingredient overlap percentage of {ingredients_percentage_match}.", icon="⚠️")
        else:
            df_filter_ingredient = df_filter_cuisine

    return df_filter_ingredient

def get_user_interactions_to_prefer(user_prev_interaction, df_filtered, df_raw_recipe, 
                                    recommendation_based_on_search_preference=True, 
                                    rating_threshold=3, n_previous=10):
    """
    Determines the user interactions to consider for recommendations based on user history 
    and current filtered data.

    Parameters:
        user_prev_interaction (pd.DataFrame): User's previous interactions with recipes.
        df_filtered (pd.DataFrame): Recipes filtered based on user preferences.
        df_raw_recipe (pd.DataFrame): The raw recipe dataset for merging recipe details.
        recommendation_based_on_search_preference (bool, optional): Whether to base recommendations 
                                                                    on the user's search preferences. 
                                                                    Defaults to True.
        rating_threshold (int, optional): Minimum rating threshold for user interactions to consider. 
                                          Defaults to 3.
        n_previous (int, optional): Number of recent interactions to consider. Defaults to 10.

    Returns:
        pd.DataFrame: Processed user interactions to prefer for recommendation.
    """
    user_interaction_to_prefer = None

    if recommendation_based_on_search_preference:
        # Find recipes already rated by the user and present in the filtered data
        recipe_id_to_prefer = set(user_prev_interaction["recipe_id"]) & set(df_filtered["id"])
        user_interaction_to_prefer = user_prev_interaction[user_prev_interaction["recipe_id"].isin(recipe_id_to_prefer)]

    # Fallback to user's entire history if no relevant interactions are found
    if user_interaction_to_prefer is None or user_interaction_to_prefer.empty:
        user_interaction_to_prefer = user_prev_interaction

    print(f"The shape of user interaction is: {user_interaction_to_prefer.shape}")

    # Filter interactions by rating threshold
    user_interaction_to_prefer = user_interaction_to_prefer[user_interaction_to_prefer["rating"] >= rating_threshold]
    # Sort by rating
    user_interaction_to_prefer = user_interaction_to_prefer.sort_values(by="rating", ascending=False)

    # Merge with raw recipe data to include recipe names
    user_interaction_to_prefer = user_interaction_to_prefer.merge(
        df_raw_recipe, left_on="recipe_id", right_on="id", how="inner"
    )

    # Select the most recent n interactions
    user_last_n_to_consider = user_interaction_to_prefer.sort_values(by="date", ascending=False).head(n_previous)

    return user_last_n_to_consider


def get_avg_ingredients_embedding(user_last_n_to_consider, embedding_dir="./data/bert_ingredients_embeddings"):
    """
    Retrieves the average BERT embedding for the user's recent recipes.

    Parameters:
        user_last_n_to_consider (pd.DataFrame): The user's last n recipe interactions to consider.
        embedding_dir (str, optional): Directory containing the BERT embeddings for ingredients. Defaults to a specific path.

    Returns:
        np.ndarray: The average ingredients embedding for the user's recent recipes.
    """
    avg_ingredients_embedding = []

    for recipe_id in user_last_n_to_consider["recipe_id"]:
        # Attempt to load the embedding for the current recipe
        try:
            with open(f"{embedding_dir}/{recipe_id}.json", "r") as f:
                ingredient_embeddings = json.load(f)["embedding"]
                avg_ingredients_embedding.append(ingredient_embeddings)
        except Exception as e:
            print(e)
            print(f"No embedding found for provided recipe ID {recipe_id}, skipping...")

    # If embeddings were successfully loaded, compute the average
    if avg_ingredients_embedding:
        avg_user_ingredients_embedding = np.mean(avg_ingredients_embedding, axis=0)
        return avg_user_ingredients_embedding
    else:
        print("No embeddings found for the provided recipes.")
        return None

def combine_user_and_search_embeddings(user_ingredients, avg_user_ingredients_embedding, get_embedding_func):
    """
    Combines the user's recipe history embeddings with the search ingredients embeddings.

    Parameters:
        user_ingredients (set): The set of ingredients searched by the user.
        avg_user_ingredients_embedding (np.ndarray): The average embedding of the user's recipe history.
        get_embedding_func (function): Function to obtain the embedding for the user-specified ingredients.

    Returns:
        np.ndarray: The combined embedding based on user history and search ingredients.
    """
    if user_ingredients:
        # Get the embedding of the ingredients the user searched for
        user_search_embedding = get_embedding_func(user_ingredients)

    if avg_user_ingredients_embedding and not np.any(np.isnan(avg_user_ingredients_embedding)) and user_ingredients:
        # Combine user history embedding with the search embedding
        avg_user_ingredients_embedding = np.mean([user_search_embedding, avg_user_ingredients_embedding], axis=0)
    elif user_ingredients:
        # If there's no valid history embedding, use the search embedding
        avg_user_ingredients_embedding = user_search_embedding

    return avg_user_ingredients_embedding

def compute_ingredient_similarity(df_filter_ingredient, avg_user_ingredients_embedding, n_recommend=10, embedding_dir="./data/bert_ingredients_embeddings"):
    """
    Computes the cosine similarity between the user's ingredient embedding and the filtered recipe ingredients.

    Parameters:
        df_filter_ingredient (pd.DataFrame): The dataset of filtered recipes.
        avg_user_ingredients_embedding (np.ndarray): The average embedding of the user's recipe history and searched ingredients.
        n_recommend (int, optional): Number of top recommended recipes to return. Defaults to 10.
        embedding_dir (str, optional): Directory containing the BERT ingredient embeddings. Defaults to a specific path.

    Returns:
        pd.DataFrame: Top recommended recipes sorted by ingredients similarity.
    """
    df_filter_ingredient["ingredients_similarity"] = np.float64(0)

    for idx, recipe_id in enumerate(df_filter_ingredient["id"]):
        # Attempt to load the embedding for the current recipe
        try:
            with open(f"{embedding_dir}/{recipe_id}.json", "r") as f:
                ingredient_embeddings = json.load(f)["embedding"]
                # Compute cosine similarity between user ingredients and recipe ingredients
                cosin_sim = cosine_similarity(
                    np.array(avg_user_ingredients_embedding).reshape(1, -1),
                    np.array(ingredient_embeddings).reshape(1, -1)
                )[0][0]
                # Store the cosine similarity in the DataFrame
                df_filter_ingredient.iloc[idx, df_filter_ingredient.columns.get_loc("ingredients_similarity")] = cosin_sim
        except Exception as e:
            print(e)
            print(f"No embedding found for provided recipe ID {recipe_id}, skipping...")

    # Sort by ingredients similarity and select the top N recipes
    recommended_recipes = df_filter_ingredient.sort_values(by="ingredients_similarity", ascending=False).head(n_recommend)

    return recommended_recipes




def filter_recipes_based_on_activity(df_filter_ingredient, user_last_n_to_consider, recommendation_based_on_activity=True, n_recommend=15, max_recipes=30):
    """
    Filters recipes based on user activity and ingredient overlap. If the filtered data is too large, it samples a subset.

    Parameters:
        df_filter_ingredient (pd.DataFrame): The dataset of filtered recipes.
        user_last_n_to_consider (pd.DataFrame): The user's last N interactions to consider.
        recommendation_based_on_activity (bool, optional): Whether to recommend based on user activity. Defaults to True.
        n_recommend (int, optional): Number of top recommended recipes to return based on ingredient overlap. Defaults to 15.
        max_recipes (int, optional): Maximum number of recipes to retain if the filtered data is too large. Defaults to 30.

    Returns:
        pd.DataFrame: Filtered and recommended recipes.
    """
    # If no user history is found and recommendations are based on activity, return empty DataFrame
    if recommendation_based_on_activity and user_last_n_to_consider.empty:
        return pd.DataFrame(columns=df_filter_ingredient.columns)

    # Focus only on user history for activity-based recommendation
    if recommendation_based_on_activity:
        # Get the union of ingredients from the user's recent activity
        user_ingredients = set.union(*user_last_n_to_consider["ingredients"].values.flatten())

        # Calculate ingredient overlap with each recipe and filter based on this overlap
        df_filter_ingredient["ingredient_overlap"] = df_filter_ingredient["ingredients"].apply(
            lambda ingredients: len(ingredients & user_ingredients) / len(user_ingredients)
        )

        # Take only top N overlapping recipes
        df_filter_ingredient = df_filter_ingredient.sort_values(by="ingredient_overlap", ascending=False).head(n_recommend)

    # If the filtered data is too large, sample a subset
    if len(df_filter_ingredient) >= max_recipes:
        print("Dataframe too big, sampling random 30 recipes.")
        df_filter_ingredient = df_filter_ingredient.sample(max_recipes)

    return df_filter_ingredient