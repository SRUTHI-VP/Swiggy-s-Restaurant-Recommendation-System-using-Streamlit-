# # # Streamlit-based Recommendation System
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import pickle

# # # Load Cleaned DataFrame
# # #cleaned_df = pd.read_csv(f"C:/Users/LENOVO/Desktop/GUVI/Project_Swiggy_AIML/cleaned_data3.csv")
# # cleaned_df = pd.read_csv(f"C:/Users/LENOVO/Desktop/GUVI/swiggy_final/cleaned_data3.csv")

# # # Load encoders, scaler, and model
# # #with open(f'C:/Users/LENOVO/Desktop/GUVI/Project_Swiggy_AIML/encoder.pkl', 'rb') as f:
# #     #encoders = pickle.load(f)

# # with open(f'C:/Users/LENOVO/Desktop/GUVI/swiggy_final/encoder.pkl', 'rb') as f:
# #     encoders = pickle.load(f)

# # with open(f'C:/Users/LENOVO/Desktop/GUVI/swiggy_final/kmeans_model.pkl', 'rb') as f:
# #     kmeans = pickle.load(f)

# # #image_url = f"C:/Users/LENOVO/Desktop/GUVI/Project_Swiggy_AIML/swiggy_image.png"
# # title = "Swiggy Restaurants Recommendation System"

# # # Create two columns
# # col1, col2 = st.columns([1, 3])

# # # Display Image in the first column
# # #with col1:
# #     #st.image(image_url, width=150)

# # # Display Title in the second column
# # with col2:
# #     st.title(f"{title}")

# # # User Input
# # city = st.selectbox('Select City', options=cleaned_df['city'].unique())
# # cuisine = st.selectbox('Select Cuisine', options=cleaned_df['cuisine'].unique())
# # cost = st.number_input('Enter Cost', min_value=0.0, step=0.1)
# # rating = st.slider("Select a Rating:", min_value=0.0, max_value=5.0, step=0.1)

# # if st.button('Get Recommendations'):

# #     # Encode city with One-Hot Encoding
# #     city_encoded = pd.DataFrame(
# #         encoders["city_encoder"].transform([[city]]),
# #         columns=encoders["city_encoder"].get_feature_names_out(["city"])
# #     )

# #     # Encode cuisine with Label Encoding
# #     cuisine_encoded = pd.DataFrame(
# #         [encoders["cuisine_encoder"].transform([cuisine])],
# #         columns=["cuisine_encoded"]
# #     )

# #     rating_scaled = pd.DataFrame(
# #         encoders["rating_scaler"].transform([[rating]]),  # Extract only the scaled cost
# #         columns=["rating_scaled"]
# #     )

# #     # Scale the cost feature
# #     cost_scaled = pd.DataFrame(
# #         encoders["cost_scaler"].transform([[cost]]),  # Extract only the scaled cost
# #         columns=["cost_scaled"]
# #     )

# #     # Combine all features using pd.concat
# #     user_input_df = pd.concat([cuisine_encoded, rating_scaled, cost_scaled, city_encoded], axis=1)

# #     # Predict the cluster for user input
# #     predicted_cluster = kmeans.predict(user_input_df)[0]
# #     print(f"Predicted Cluster: {predicted_cluster}")

# #     # Step 4: Get Recommendations
# #     # Filter encoded data based on the cluster and map back to cleaned data
# #     cluster_indices = (kmeans.labels_ == predicted_cluster)
# #     result = cleaned_df.loc[cluster_indices]


# #     recommended_restaurants = result[["name", "link"]].rename(
# #         columns={
# #             "name": "Restaurant_Name",
# #             "link": "Link"
# #         }
# #     )

# #     recommended_restaurants.reset_index(drop=True,inplace=True)

# #     st.subheader('Recommended Restaurants:')
# #     st.table(recommended_restaurants.iloc[1:11,:])


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# st.set_page_config(page_title="Swiggy Restaurant Recommender", layout="wide")

# # -----------------------------
# # Load Data and Models
# # -----------------------------
# DATA_PATH = r"C:/Users/LENOVO/Desktop/GUVI/swiggy_final/cleaned_data3.csv"
# ENCODER_PATH = r"C:/Users/LENOVO/Desktop/GUVI/swiggy_final/encoder.pkl"
# MODEL_PATH = r"C:/Users/LENOVO/Desktop/GUVI/swiggy_final/kmeans_model.pkl"

# cleaned_df = pd.read_csv(DATA_PATH)

# with open(ENCODER_PATH, "rb") as f:
#     encoders = pickle.load(f)

# with open(MODEL_PATH, "rb") as f:
#     kmeans = pickle.load(f)

# # -----------------------------
# # Basic Cleaning
# # -----------------------------
# cleaned_df = cleaned_df.dropna(subset=["name", "city", "cuisine", "rating", "cost"]).copy()

# cleaned_df["city"] = cleaned_df["city"].astype(str).str.strip()
# cleaned_df["cuisine"] = cleaned_df["cuisine"].astype(str).str.strip()
# cleaned_df["name"] = cleaned_df["name"].astype(str).str.strip()
# cleaned_df["rating"] = pd.to_numeric(cleaned_df["rating"], errors="coerce")
# cleaned_df["cost"] = pd.to_numeric(cleaned_df["cost"], errors="coerce")

# cleaned_df = cleaned_df.dropna(subset=["rating", "cost"]).copy()

# # -----------------------------
# # Title
# # -----------------------------
# st.title("Swiggy Restaurants Recommendation System")
# st.write("Select your preferences to get restaurant recommendations.")

# # -----------------------------
# # Inputs
# # -----------------------------
# city_options = sorted(cleaned_df["city"].dropna().unique().tolist())
# selected_city = st.selectbox("Select City", city_options)

# city_df = cleaned_df[cleaned_df["city"] == selected_city].copy()

# cuisine_options = sorted(city_df["cuisine"].dropna().unique().tolist())
# selected_cuisine = st.selectbox("Select Cuisine", cuisine_options)

# min_cost = float(cleaned_df["cost"].min())
# max_cost = float(cleaned_df["cost"].max())

# selected_cost = st.number_input(
#     "Enter Maximum Cost",
#     min_value=min_cost,
#     max_value=max_cost,
#     value=min(300.0, max_cost),
#     step=50.0
# )

# selected_rating = st.slider(
#     "Select Minimum Rating",
#     min_value=0.0,
#     max_value=5.0,
#     value=3.5,
#     step=0.1
# )

# top_n = st.slider("Number of Recommendations", 5, 20, 10)

# # -----------------------------
# # Helper: build model input
# # -----------------------------
# def build_user_input(city, cuisine, rating, cost):
#     city_encoded = pd.DataFrame(
#         encoders["city_encoder"].transform([[city]]),
#         columns=encoders["city_encoder"].get_feature_names_out(["city"])
#     )

#     cuisine_encoded = pd.DataFrame(
#         [encoders["cuisine_encoder"].transform([cuisine])],
#         columns=["cuisine_encoded"]
#     )

#     rating_scaled = pd.DataFrame(
#         encoders["rating_scaler"].transform([[rating]]),
#         columns=["rating_scaled"]
#     )

#     cost_scaled = pd.DataFrame(
#         encoders["cost_scaler"].transform([[cost]]),
#         columns=["cost_scaled"]
#     )

#     user_input_df = pd.concat(
#         [cuisine_encoded, rating_scaled, cost_scaled, city_encoded],
#         axis=1
#     )

#     return user_input_df

# # -----------------------------
# # Recommend
# # -----------------------------
# if st.button("Get Recommendations"):
#     try:
#         user_input_df = build_user_input(
#             selected_city,
#             selected_cuisine,
#             selected_rating,
#             selected_cost
#         )

#         predicted_cluster = kmeans.predict(user_input_df)[0]
#         st.info(f"Predicted Cluster: {predicted_cluster}")

#         cluster_indices = (kmeans.labels_ == predicted_cluster)
#         result = cleaned_df.loc[cluster_indices].copy()

#         # Filter by actual user preferences
#         result = result[result["city"] == selected_city]
#         result = result[result["rating"] >= selected_rating]
#         result = result[result["cost"] <= selected_cost]

#         result = result[
#             result["cuisine"].str.contains(selected_cuisine, case=False, na=False)
#         ]

#         # Fallback if strict filter returns nothing
#         if result.empty:
#             result = cleaned_df.loc[cluster_indices].copy()
#             result = result[result["city"] == selected_city]

#         if result.empty:
#             result = cleaned_df[
#                 (cleaned_df["city"] == selected_city) &
#                 (cleaned_df["rating"] >= selected_rating) &
#                 (cleaned_df["cost"] <= selected_cost) &
#                 (cleaned_df["cuisine"].str.contains(selected_cuisine, case=False, na=False))
#             ].copy()

#         if result.empty:
#             st.warning("No matching restaurants found for the selected filters.")
#         else:
#             result = result.sort_values(
#                 by=["rating", "rating_count", "cost"],
#                 ascending=[False, False, True]
#             )

#             recommended_restaurants = result[
#                 ["name", "city", "cuisine", "rating", "rating_count", "cost", "link", "address"]
#             ].rename(
#                 columns={
#                     "name": "Restaurant_Name",
#                     "city": "City",
#                     "cuisine": "Cuisine",
#                     "rating": "Rating",
#                     "rating_count": "Rating_Count",
#                     "cost": "Cost",
#                     "link": "Link",
#                     "address": "Address"
#                 }
#             ).drop_duplicates(subset=["Restaurant_Name", "Link"])

#             recommended_restaurants = recommended_restaurants.head(top_n).reset_index(drop=True)

#             st.subheader("Recommended Restaurants")
#             st.dataframe(recommended_restaurants, use_container_width=True)
            

#     except Exception as e:
#         st.error(f"Error: {e}")


import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Swiggy Restaurant Recommender", layout="wide")

DATA_PATH = r"C:/Users/LENOVO/Desktop/GUVI/swiggy_final/cleaned_data3.csv"
ENCODER_PATH = r"C:/Users/LENOVO/Desktop/GUVI/swiggy_final/encoder.pkl"
MODEL_PATH = r"C:/Users/LENOVO/Desktop/GUVI/swiggy_final/kmeans_model.pkl"

cleaned_df = pd.read_csv(DATA_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    kmeans = pickle.load(f)

cleaned_df = cleaned_df.dropna(subset=["name", "city", "cuisine", "rating", "cost", "link"]).copy()

cleaned_df["city"] = cleaned_df["city"].astype(str).str.strip()
cleaned_df["cuisine"] = cleaned_df["cuisine"].astype(str).str.strip()
cleaned_df["name"] = cleaned_df["name"].astype(str).str.strip()
cleaned_df["link"] = cleaned_df["link"].astype(str).str.strip()
cleaned_df["rating"] = pd.to_numeric(cleaned_df["rating"], errors="coerce")
cleaned_df["cost"] = pd.to_numeric(cleaned_df["cost"], errors="coerce")
cleaned_df["rating_count"] = pd.to_numeric(cleaned_df["rating_count"], errors="coerce")

cleaned_df = cleaned_df.dropna(subset=["rating", "cost"]).copy()

cleaned_df = cleaned_df[cleaned_df["link"].str.startswith("http", na=False)]
cleaned_df = cleaned_df[~cleaned_df["link"].str.contains(r"\.\.\.", regex=True, na=False)]

st.title("Swiggy Restaurants Recommendation System")
st.write("Select your preferences to get restaurant recommendations.")

city_options = sorted(cleaned_df["city"].dropna().unique().tolist())
selected_city = st.selectbox("Select City", city_options)

city_df = cleaned_df[cleaned_df["city"] == selected_city].copy()
cuisine_options = sorted(city_df["cuisine"].dropna().unique().tolist())
selected_cuisine = st.selectbox("Select Cuisine", cuisine_options)

min_cost = float(cleaned_df["cost"].min())
max_cost = float(cleaned_df["cost"].max())

selected_cost = st.number_input(
    "Enter Maximum Cost",
    min_value=min_cost,
    max_value=max_cost,
    value=min(300.0, max_cost),
    step=50.0
)

selected_rating = st.slider(
    "Select Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=3.5,
    step=0.1
)

top_n = st.slider("Number of Recommendations", 5, 20, 10)

def build_user_input(city, cuisine, rating, cost):
    city_encoded = pd.DataFrame(
        encoders["city_encoder"].transform([[city]]),
        columns=encoders["city_encoder"].get_feature_names_out(["city"])
    )

    cuisine_encoded = pd.DataFrame(
        [encoders["cuisine_encoder"].transform([cuisine])],
        columns=["cuisine_encoded"]
    )

    rating_scaled = pd.DataFrame(
        encoders["rating_scaler"].transform([[rating]]),
        columns=["rating_scaled"]
    )

    cost_scaled = pd.DataFrame(
        encoders["cost_scaler"].transform([[cost]]),
        columns=["cost_scaled"]
    )

    user_input_df = pd.concat(
        [cuisine_encoded, rating_scaled, cost_scaled, city_encoded],
        axis=1
    )

    return user_input_df

if st.button("Get Recommendations"):
    try:
        user_input_df = build_user_input(
            selected_city,
            selected_cuisine,
            selected_rating,
            selected_cost
        )

        predicted_cluster = kmeans.predict(user_input_df)[0]
        st.info(f"Predicted Cluster: {predicted_cluster}")

        cluster_indices = (kmeans.labels_ == predicted_cluster)
        result = cleaned_df.loc[cluster_indices].copy()

        result = result[result["city"] == selected_city]
        result = result[result["rating"] >= selected_rating]
        result = result[result["cost"] <= selected_cost]
        result = result[
            result["cuisine"].str.contains(selected_cuisine, case=False, na=False)
        ]

        if result.empty:
            result = cleaned_df.loc[cluster_indices].copy()
            result = result[result["city"] == selected_city]

        if result.empty:
            result = cleaned_df[
                (cleaned_df["city"] == selected_city) &
                (cleaned_df["rating"] >= selected_rating) &
                (cleaned_df["cost"] <= selected_cost) &
                (cleaned_df["cuisine"].str.contains(selected_cuisine, case=False, na=False))
            ].copy()

        if result.empty:
            st.warning("No matching restaurants found for the selected filters.")
        else:
            result = result.sort_values(
                by=["rating", "rating_count", "cost"],
                ascending=[False, False, True]
            )

            recommended_restaurants = result[
                ["name", "city", "cuisine", "rating", "rating_count", "cost", "link", "address"]
            ].rename(
                columns={
                    "name": "Restaurant_Name",
                    "city": "City",
                    "cuisine": "Cuisine",
                    "rating": "Rating",
                    "rating_count": "Rating_Count",
                    "cost": "Cost",
                    "link": "Link",
                    "address": "Address"
                }
            ).drop_duplicates(subset=["Restaurant_Name", "Link"])

            # recommended_restaurants = recommended_restaurants.head(top_n).reset_index(drop=True)

            # st.subheader("Recommended Restaurants")
            # st.dataframe(
            #     recommended_restaurants,
            #     use_container_width=True,
            #     hide_index=True,
            #     column_config={
            #         "Link": st.column_config.LinkColumn(
            #             "Restaurant Link",
            #             display_text="Open Link"
            #         )
            #     }
            # )
            st.subheader("Recommended Restaurants")

            display_df = recommended_restaurants[["Restaurant_Name", "City", "Cuisine", "Rating", "Rating_Count", "Cost", "Address"]
]

            st.dataframe(display_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error: {e}")