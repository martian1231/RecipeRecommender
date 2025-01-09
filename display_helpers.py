import re
import demoji

def display_user_history_and_emojis(
    df_raw_recipe, df_raw_user_inter, col1, recommend_function, emoji_file_path="./data/emojis_list.txt", st= None
):
    """
    Displays user history and emojis in a grid format.

    Args:
        df_raw_recipe (pd.DataFrame): The dataframe containing recipe data.
        df_raw_user_inter (pd.DataFrame): The dataframe containing user interaction data.
        col1: Streamlit column to display content.
        recommend_function (function): Function to generate recommendations.
        emoji_file_path (str): Path to the file containing emoji list.
    """
    with col1:
        # Filtered user history display
        st.markdown(f"#### Refined user history (user_id: {st.session_state.user_id} )")
        strict_user_preferences = {
            "cuisine": {},
            "diet_type": {},
            "time_limit_min": 0,
            "available_ingredients": {},
            "selected_user_id": st.session_state.user_id,
        }

        # Recommendations caching
        if "col1_recommend" not in st.session_state:
            recommendations = recommend_function(
                df_raw_recipe,
                df_raw_user_inter,
                strict_user_preferences,
                recommendation_based_on_search_preference=False,
                return_history=True,
                st=st,
            )
            st.session_state.col1_recommend = recommendations

            # Load emojis from file if not already loaded
            with open(emoji_file_path, "r", encoding="utf-8") as file:
                emojis_list = [line.strip() for line in file]
                st.session_state.emojis_list = emojis_list
        else:
            recommendations = st.session_state.col1_recommend
            emojis_list = st.session_state.emojis_list

        # Display the recommendations in a dataframe
        df_show = recommendations[["recipe_id", "name", "rating", "review", "date"]]
        st.dataframe(df_show)

        # Create an HTML table for displaying emojis in a grid
        html_table = "<table style='width: 100%; text-align: center; font-size: 2em;'>"

        # Set the number of columns for the grid
        num_columns = 5  # Number of columns in the grid

        # Divide emojis into rows based on the number of columns
        rows = [
            emojis_list[i : i + num_columns]
            for i in range(0, len(emojis_list), num_columns)
        ]

        for row in rows:
            html_table += "<tr>"
            for emoji in row:
                html_table += f"<td>{emoji}</td>"
            html_table += "</tr>"

        html_table += "</table>"

        # Display the HTML table
        st.markdown(html_table, unsafe_allow_html=True)



def display_recipe_recommendation_form(col2, df_raw_recipe, df_raw_user_inter, diet_types, cuisines, normalize_ingredient, recommend, st):
    """
    Displays a recipe recommendation form and processes user input to generate recommendations.

    Args:
        col2: Streamlit column where the form and recommendations will be displayed.
        df_raw_recipe: DataFrame containing raw recipe data.
        df_raw_user_inter: DataFrame containing user interaction data.
        diet_types: List of available diet types.
        cuisines: List of available cuisines.
        normalize_ingredient: Function to normalize ingredient list.
        recommend: Function to generate recipe recommendations.
    """

    def process_time_limit(selected_option):
        """Processes the time limit based on user selection."""
        if selected_option == "Manual Input":
            manual_input = st.text_input("Enter time limit (in mins):", value="15")
            return int(manual_input) if manual_input else None
        return selected_option

    def process_user_preferences():
        """Processes user preferences from the form."""
        time_limit = process_time_limit(selected_option)
        diet_pref = {selected_diet_type} if selected_diet_type != "all" else {}
        cuisine_pref = {selected_cuisine_type} if selected_cuisine_type != "all" else {}
        user_ingredients = re.findall(r"[a-zA-Z]+", ingredient_list)
        emoji_as_text = demoji.findall(ingredient_list)
        user_ingredients = set(user_ingredients).union(set(emoji_as_text.values()))
        print("user_ingredients", user_ingredients)
        normalized_ingredients = normalize_ingredient(user_ingredients)

        return {
            "cuisine": cuisine_pref,
            "diet_type": diet_pref,
            "time_limit_min": max(0, time_limit),
            "available_ingredients": normalized_ingredients,
            "selected_user_id": st.session_state.user_id
        }

    def display_recommendations(recommendations, st):
        """Displays the list of recipe recommendations."""
        n_recommend = len(recommendations)
        if n_recommend:
            cols = st.columns(n_recommend)
            for idx, col in enumerate(cols):
                with col:
                    st.markdown(f"**Recipe: {recommendations.name.iloc[idx]}**")
                    st.text(f"TTC: {recommendations.minutes.iloc[idx]} mins.")
                    st.image("./data/background.png", width=50)
                    steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(recommendations.steps.iloc[idx])])
                    st.markdown("**Steps to Prepare:**")
                    st.markdown(steps)
        else:
            st.text("No recommendation for current preferences provided.")


    with col2:
        with st.form(key='recipe_form'):
            options = [10, 20, 30, 40, 50]
            selected_option = st.selectbox("Choose the cooking time limit for your recipe selection:", options + ["Manual Input"], index=2)
            selected_diet_type = st.selectbox(' Select your preferred diet type to filter recipes according to your dietary restrictions:', diet_types, index=1)
            selected_cuisine_type = st.selectbox('Select your preferred cuisine to filter recipes based on regional or cultural flavors:', cuisines, index=3)
            ingredient_list = st.text_area("Provide list of ingredients required for preparing a specific recipe:", value="🧅, 🍅, lentils").strip()
            ingredients_percentage_match = st.slider("Select threshold that filters recipes based on the percentage of ingredient match provided in the ingredients list:", min_value=0.0, max_value=1.0, step=0.05, value=1.0)
            submitted = st.form_submit_button("Search Recipes")

        if submitted:
            errors = []
            if not process_time_limit(selected_option):
                errors.append("Time Limit cannot be empty.")
            if not selected_diet_type:
                errors.append("Please select a diet type.")
            if not selected_cuisine_type:
                errors.append("Please select a cuisine.")
            if not ingredient_list:
                errors.append("Ingredients list cannot be empty.")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                user_preferences = process_user_preferences()
                recommendations = recommend(
                    df_raw_recipe=df_raw_recipe,
                    df_raw_user_inter=df_raw_user_inter,
                    strict_user_preferences=user_preferences,
                    ingredients_percentage_match=ingredients_percentage_match,
                    recommendation_based_on_search_preference=True,
                    return_history=False,
                    n_recommend=5,
                    st=st
                )
                display_recommendations(recommendations, st)



def display_recommendations_based_on_past_search(col3, df_raw_recipe, df_raw_user_inter, recommend, st):
    """
    Display recipe recommendations based on past searches in the specified Streamlit column.

    Args:
        col3: The Streamlit column to display the recommendations.
        df_raw_recipe: DataFrame containing recipe information.
        df_raw_user_inter: DataFrame containing user interaction data.
    """
    with col3:
        st.markdown(f"#### Recommendations based on previous recipe interactions:")

        strict_user_preferences = {
            "cuisine": {},
            "diet_type": {},
            "time_limit_min": 0,
            "available_ingredients": {},
            "selected_user_id": st.session_state.user_id
        }

        # Retrieve or calculate recommendations
        recommendations = get_or_calculate_recommendations(
            df_raw_recipe, df_raw_user_inter, strict_user_preferences, recommend, st
        )

        # Display the recommendations
        display_recommendations(recommendations, st)

def get_or_calculate_recommendations(df_raw_recipe, df_raw_user_inter, user_preferences, recommend, st):
    """
    Retrieve recommendations from session state or calculate them if not already stored.

    Args:
        df_raw_recipe: DataFrame containing recipe information.
        df_raw_user_inter: DataFrame containing user interaction data.
        user_preferences: Dictionary of strict user preferences.

    Returns:
        DataFrame of recommendations.
    """
    if "col3_recommend" not in st.session_state:
        recommendations = recommend(
            df_raw_recipe=df_raw_recipe,
            df_raw_user_inter=df_raw_user_inter,
            strict_user_preferences=user_preferences,
            recommendation_based_on_search_preference=False,
            return_history=False,
            n_recommend=3,
            recommendation_based_on_activity=True,
            st=st
        )
        st.session_state.col3_recommend = recommendations
    else:
        recommendations = st.session_state.col3_recommend

    return recommendations

def display_recommendations(recommendations, st):
    """
    Display the list of recommendations in a grid layout.

    Args:
        recommendations: DataFrame of recipe recommendations.
    """
    n_recommend = len(recommendations)

    if n_recommend:
        cols = st.columns(n_recommend)

        for idx, col in enumerate(cols):
            with col:
                st.markdown(f"**Recipe: {recommendations.name.iloc[idx]}**")
                st.text(f"TTC: {recommendations.minutes.iloc[idx]} mins.")
                st.image(
                    "./data/background.png",
                    width=50
                )

                # Display steps using Markdown
                st.markdown("**Steps to Prepare:**")
                steps = "\n".join(
                    [f"{i+1}. {step}" for i, step in enumerate(recommendations.steps.iloc[idx])]
                )
                st.markdown(steps)
    else:
        st.text("No recommendations available for the current preferences")
