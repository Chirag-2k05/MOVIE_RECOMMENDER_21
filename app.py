import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Load dataset
@st.cache
def load_data():
    ratings = pd.read_csv(ratings.csv)
    movies = pd.read_csv(movies.csv)
    data = pd.merge(ratings, movies, on='movieId')
    user_item_matrix = data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    return data, user_item_matrix, movies

data, user_item_matrix, movies = load_data()

# Matrix Factorization using SVD
def svd_recommend(user_id, user_item_matrix, num_recommendations=5):
    U, sigma, Vt = svds(user_item_matrix, k=50)
    sigma = np.diag(sigma)
    svd_predictions = np.dot(np.dot(U, sigma), Vt)
    predicted_ratings = pd.DataFrame(svd_predictions, columns=user_item_matrix.columns)
    user_row_number = user_id - 1
    sorted_user_predictions = predicted_ratings.iloc[user_row_number].sort_values(ascending=False)
    user_data = data[data.userId == user_id]
    recommendations = (movies[~movies['movieId'].isin(user_data['movieId'])]
                       .merge(pd.DataFrame(sorted_user_predictions).reset_index(), on='movieId')
                       .rename(columns={user_row_number: 'Predictions'})
                       .sort_values('Predictions', ascending=False)
                       .iloc[:num_recommendations, :-1])
    return recommendations

# Streamlit UI
st.title("Movie Recommendation System")

st.sidebar.header("User Input")
user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=len(user_item_matrix), step=1)
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, step=1, value=5)

if st.sidebar.button("Get Recommendations"):
    recommendations = svd_recommend(user_id, user_item_matrix, num_recommendations)
    st.write(f"Top {num_recommendations} movie recommendations for User {user_id}:")
    st.table(recommendations)
