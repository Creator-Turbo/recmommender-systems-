from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the models
movie_list = pickle.load(open('model/movie_list.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

# Helper function to get recommendations
def recommend(movie):
    # Get the index of the movie
    movie_index = movie_list[movie_list['title'] == movie].index[0]
    # Calculate similarity
    sim_scores = list(enumerate(similarity[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_movies = [movie_list['title'][i[0]] for i in sim_scores[1:6]]
    return recommended_movies

@app.route('/')
def index():
    return render_template('index.html', movies=movie_list['title'].tolist())

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    movie = request.form['movie']
    recommendations = recommend(movie)
    return render_template('index.html', movies=movie_list['title'].tolist(), recommendations=recommendations, selected_movie=movie)

if __name__ == '__main__':
    app.run(debug=True)
