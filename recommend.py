import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from utils import get_embedding, filter_recipes_by_time
from utils import filter_recipes_by_diet, filter_recipes_by_cuisine, filter_recipes_by_ingredients, get_user_interactions_to_prefer, get_avg_ingredients_embedding, combine_user_and_search_embeddings, compute_ingredient_similarity, filter_recipes_based_on_activity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def recommend_recipes(df_raw_recipe, df_raw_user_inter,
              strict_user_preferences, st, recommendation_based_on_search_preference=False,
              return_history= False, n_recommend=5, n_previous=5,
              ingredients_percentage_match=0.2, recommendation_based_on_activity= False):

    time_limit_pref = max(0 ,strict_user_preferences["time_limit_min"])
    
    diet_pref = strict_user_preferences["diet_type"]
    cuisine_pref = strict_user_preferences["cuisine"]
    
    user_ingredients = strict_user_preferences["available_ingredients"]

    # test user id from dataset, because we need previous data
    test_uid = int(strict_user_preferences["selected_user_id"])
    
    diet_pref_len = len(diet_pref)
    cuisine_pref_len = len(cuisine_pref)
    
    df_filter_time = filter_recipes_by_time(df_raw_recipe, time_limit_pref)
    
    if not len(df_filter_time):
        print(f"No recipe found which can be prepared in {time_limit_pref} minutes")
        return df_filter_time
    
    df_filter_diet = filter_recipes_by_diet(df_filter_time, diet_pref=diet_pref, diet_pref_len=diet_pref_len)
    
    if not len(df_filter_diet):
        print(f"No recipe found with diet preference {diet_pref}")
        return df_filter_diet

    df_filter_cuisine = filter_recipes_by_cuisine(df_filter_diet, cuisine_pref=cuisine_pref, cuisine_pref_len=cuisine_pref_len)
    
    
    if not len(df_filter_cuisine):
        print(f"No recipe found with cuisine preference {cuisine_pref}")
        return df_filter_cuisine
    
    df_filter_ingredient = filter_recipes_by_ingredients(df_filter_cuisine, user_ingredients=user_ingredients, ingredients_percentage_match=ingredients_percentage_match, st= st)
    
    
    if not len(df_filter_ingredient):
        print(f"No recipe found with ingredients preference {user_ingredients}")
    
    df_filtered = df_filter_ingredient
        
    # user previous interactions
    user_prev_interaction = df_raw_user_inter[df_raw_user_inter.user_id == test_uid]

    user_last_n_to_consider = get_user_interactions_to_prefer(user_prev_interaction, df_filtered, df_raw_recipe, 
                                    recommendation_based_on_search_preference=recommendation_based_on_search_preference, 
                                    rating_threshold=3, n_previous=n_previous)
    
    if return_history:
        return user_last_n_to_consider
    
    # remove all the recipes which user has already rated/ tried so we can recommend fresh recipes
    df_filter_ingredient = df_filter_ingredient[~df_filter_ingredient.id.isin(user_last_n_to_consider.recipe_id)]

    df_filter_ingredient = filter_recipes_based_on_activity(df_filter_ingredient, user_last_n_to_consider, recommendation_based_on_activity=True, n_recommend=15, max_recipes=30)

    avg_user_ingredients_embedding = get_avg_ingredients_embedding(user_last_n_to_consider, embedding_dir="./data/bert_ingredients_embeddings/bert_ingredients_embeddings")

    avg_user_ingredients_embedding = combine_user_and_search_embeddings(user_ingredients, avg_user_ingredients_embedding, get_embedding)

    recommended_recipes = compute_ingredient_similarity(df_filter_ingredient, avg_user_ingredients_embedding, n_recommend=n_recommend, embedding_dir="./data/bert_ingredients_embeddings/bert_ingredients_embeddings")

    return recommended_recipes