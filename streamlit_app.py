import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os

from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Swiggy Restaurant Recommendation System",
    page_icon="🍴",
    layout="wide"
)


# ==========================================================
# LOAD CLEANED DATA
# ==========================================================

@st.cache_data
def load_cleaned_data():

    df = pd.read_csv(
        "data/cleaned_data.csv"
    )

    return df


# ==========================================================
# LOAD ENCODED DATA
# ==========================================================

@st.cache_data
def load_encoded_data():

    encoded_df = pd.read_csv(
        "data/encoded_data.csv",
        index_col=0
    )

    return encoded_df


# ==========================================================
# LOAD ENCODERS
# ==========================================================

@st.cache_resource
def load_encoders():

    with open(
        "models/encoder.pkl",
        "rb"
    ) as file:

        preprocessing_objects = pickle.load(file)

    city_encoder = (
        preprocessing_objects["city_encoder"]
    )

    cuisine_encoder = (
        preprocessing_objects["cuisine_encoder"]
    )

    scaler = (
        preprocessing_objects["scaler"]
    )

    return (
        city_encoder,
        cuisine_encoder,
        scaler
    )


# ==========================================================
# LOAD EVERYTHING
# ==========================================================

df = load_cleaned_data()

encoded_df = load_encoded_data()

(
    city_encoder,
    cuisine_encoder,
    scaler
) = load_encoders()


# ==========================================================
# CHECK DATA ALIGNMENT
# ==========================================================

if not df.index.equals(
    encoded_df.index
):

    st.error(
        "Error: cleaned data and encoded data "
        "indices do not match."
    )

    st.stop()


# ==========================================================
# CHECK FEATURE COUNT
# ==========================================================

if encoded_df.shape[1] != 947:

    st.warning(
        f"Expected 950 encoded features, "
        f"but found {encoded_df.shape[1]}."
    )


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "name",
    "address",
    "city",
    "rating",
    "rating_count",
    "cost",
    "cuisine"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        f"Missing columns in cleaned_data.csv: "
        f"{missing_columns}"
    )

    st.stop()


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "🍴 Swiggy Restaurant Recommendation System"
)

st.write(
    "Find restaurants based on your city, cuisine, "
    "rating and budget preferences."
)

st.markdown("---")


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(
    "📌 About the System"
)

st.sidebar.write(
    """
    This recommendation system uses:

    • Data preprocessing
    • One-Hot Encoding
    • Multi-Label Cuisine Encoding
    • Feature Scaling
    • Cosine Similarity

    Enter your preferences to get
    restaurant recommendations.
    """
)


# ==========================================================
# USER INPUT
# ==========================================================

st.subheader(
    "🔍 Enter Your Preferences"
)

col1, col2 = st.columns(2)


# ==========================================================
# CITY AND CUISINE
# ==========================================================

with col1:

    city = st.selectbox(
        "🏙️ Select City",

        sorted(
            df["city"]
            .dropna()
            .unique()
        )
    )


    cuisine = st.selectbox(
        "🍛 Select Cuisine",

        sorted(
            df["cuisine"]
            .dropna()
            .unique()
        )
    )


# ==========================================================
# RATING AND COST
# ==========================================================

with col2:

    min_rating = st.slider(
        "⭐ Minimum Rating",

        min_value=1.0,

        max_value=5.0,

        value=3.5,

        step=0.1
    )


    max_cost = st.slider(
        "💰 Maximum Cost",

        min_value=1,

        max_value=int(
            df["cost"].max()
        ),

        value=500,

        step=50
    )


st.markdown("---")


# ==========================================================
# RECOMMEND BUTTON
# ==========================================================

if st.button(
    "🔍 Recommend Restaurants",
    type="primary"
):

    # ======================================================
    # STEP 1 — FILTER RESTAURANTS
    # ======================================================

    filtered_df = df[
        (df["city"] == city) &
        (df["rating"] >= min_rating) &
        (df["cost"] <= max_cost)
    ].copy()


    # ======================================================
    # STEP 2 — CUISINE MATCHING
    # ======================================================

    selected_cuisines = set(
        item.strip().lower()
        for item in cuisine.split(",")
        if item.strip()
    )


    def cuisine_match(cuisine_value):

        restaurant_cuisines = set(
            item.strip().lower()
            for item in str(
                cuisine_value
            ).split(",")
            if item.strip()
        )

        return bool(
            selected_cuisines.intersection(
                restaurant_cuisines
            )
        )


    filtered_df = filtered_df[
        filtered_df["cuisine"].apply(
            cuisine_match
        )
    ].copy()


    # ======================================================
    # STEP 3 — CHECK RESULTS
    # ======================================================

    if filtered_df.empty:

        st.warning(
            "No restaurants found matching "
            "all your preferences."
        )

        st.info(
            "Try lowering the minimum rating "
            "or increasing the maximum cost."
        )

        st.stop()


    # ======================================================
    # STEP 4 — CREATE USER QUERY
    # ======================================================

    query_rating = min_rating

    query_rating_count = (
        df["rating_count"].median()
    )

    query_cost = max_cost


    numerical_query = np.array(
        [[
            query_rating,
            query_rating_count,
            query_cost
        ]]
    )


    # ======================================================
    # STEP 5 — SCALE NUMERICAL FEATURES
    # ======================================================

    numerical_scaled = (
        scaler.transform(
            numerical_query
        )
    )


    # ======================================================
    # STEP 6 — ENCODE CITY
    # ======================================================

    city_encoded = city_encoder.transform(
        [[city]]
    )


    # Convert to NumPy array

    city_encoded = np.asarray(
        city_encoded
    )


    # ======================================================
    # STEP 7 — ENCODE CUISINE
    # ======================================================

    cuisine_list = [
        item.strip()
        for item in cuisine.split(",")
        if item.strip()
    ]


    cuisine_encoded = (
        cuisine_encoder.transform(
            [cuisine_list]
        )
    )


    cuisine_encoded = np.asarray(
        cuisine_encoded
    )


    # ======================================================
    # STEP 8 — COMBINE FEATURES
    # ======================================================

    query_vector = np.hstack(
        [
            numerical_scaled,
            city_encoded,
            cuisine_encoded
        ]
    )


    # ======================================================
    # STEP 9 — CHECK QUERY DIMENSION
    # ======================================================

    if query_vector.shape[1] != encoded_df.shape[1]:

        st.error(
            "Feature mismatch between the "
            "user query and encoded data."
        )

        st.write(
            "Query features:",
            query_vector.shape[1]
        )

        st.write(
            "Encoded data features:",
            encoded_df.shape[1]
        )

        st.stop()


    # ======================================================
    # STEP 10 — GET MATCHING INDICES
    # ======================================================

    matching_indices = (
        filtered_df.index
    )


    # ======================================================
    # STEP 11 — GET CORRESPONDING ENCODED DATA
    # ======================================================

    matching_encoded = (
        encoded_df.loc[
            matching_indices
        ]
    )


    # Convert to NumPy

    matching_encoded_array = (
        matching_encoded.to_numpy(
            dtype=float
        )
    )


    # ======================================================
    # STEP 12 — COSINE SIMILARITY
    # ======================================================

    similarity_scores = (
        cosine_similarity(
            query_vector,
            matching_encoded_array
        )[0]
    )


    # ======================================================
    # STEP 13 — ADD SIMILARITY SCORE
    # ======================================================

    recommendations = (
        filtered_df.copy()
    )


    recommendations[
        "similarity_score"
    ] = similarity_scores


    # ======================================================
    # STEP 14 — SORT BY SIMILARITY
    # ======================================================

    recommendations = (
        recommendations
        .sort_values(
            by="similarity_score",
            ascending=False
        )
    )


    # ======================================================
    # STEP 15 — REMOVE DUPLICATES
    # ======================================================

    if "id" in recommendations.columns:

        recommendations = (
            recommendations
            .drop_duplicates(
                subset=["id"]
            )
        )


    # ======================================================
    # STEP 16 — TOP 10
    # ======================================================

    recommendations = (
        recommendations
        .head(10)
    )


    # ======================================================
    # STEP 17 — DISPLAY RESULTS
    # ======================================================

    st.markdown("---")

    st.subheader(
        "🍽️ Recommended Restaurants"
    )


    st.success(
        f"Found {len(filtered_df)} restaurants "
        f"matching your preferences. "
        f"Showing the top {len(recommendations)}."
    )


    # ======================================================
    # SELECT DISPLAY COLUMNS
    # ======================================================

    display_df = recommendations[
        [
            "name",
            "address",
            "city",
            "rating",
            "rating_count",
            "cost",
            "cuisine",
            "similarity_score"
        ]
    ].copy()


    # ======================================================
    # RENAME COLUMNS
    # ======================================================

    display_df.columns = [
        "Restaurant",
        "Address",
        "City",
        "Rating",
        "Rating Count",
        "Cost (₹)",
        "Cuisine",
        "Similarity Score"
    ]


    # ======================================================
    # ROUND VALUES
    # ======================================================

    display_df[
        "Rating"
    ] = display_df[
        "Rating"
    ].round(1)


    display_df[
        "Rating Count"
    ] = display_df[
        "Rating Count"
    ].round(0)


    display_df[
        "Cost (₹)"
    ] = display_df[
        "Cost (₹)"
    ].round(0)


    display_df[
        "Similarity Score"
    ] = display_df[
        "Similarity Score"
    ].round(3)


    # ======================================================
    # DISPLAY DATAFRAME
    # ======================================================

    st.dataframe(
        display_df,

        use_container_width=True,

        hide_index=True
    )