import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="Recipe Recommendations",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('<h1 style="text-align: center;">Welcome to the Food Recommendation App</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Explore recipes tailored to your preferences!</p>', unsafe_allow_html=True)

import pandas as pd
import pickle
import requests
import re
from nltk.stem import WordNetLemmatizer
import nltk
from display_helpers import display_user_history_and_emojis, display_recipe_recommendation_form, display_recommendations_based_on_past_search
from recommend import recommend_recipes
from utils import normalize_ingredient
from data_helpers import load_data, download_data
from constants import diet_types, cuisines

# Download data
download_data(st)
# Load data
df_raw_recipe, df_raw_user_inter = load_data(st)

# Set up columns
col1, col2, col3 = st.columns(3)

# Initialize session state for user ID
if "user_id" not in st.session_state:
    user_id = df_raw_user_inter.user_id.sample(1).iloc[0]
    st.session_state.user_id = user_id

# If user ID comes from query parameters, update session state
if "user_id" in st.query_params:
    st.session_state.user_id = st.query_params["user_id"]


display_user_history_and_emojis(
    df_raw_recipe, df_raw_user_inter, col1, recommend_recipes, emoji_file_path="./data/emojis_list.txt", st= st)

display_recipe_recommendation_form(col2, df_raw_recipe, df_raw_user_inter, diet_types, cuisines, normalize_ingredient, recommend_recipes, st)


display_recommendations_based_on_past_search(col3, df_raw_recipe, df_raw_user_inter, recommend_recipes, st)
