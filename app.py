import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

print("⏳ Loading the database...")

# ---------------------------------------------------------
# STEP 1: LOAD AND PREPARE DATA
# ---------------------------------------------------------
df = None
cosine_sim = None
csv_file = 'tmdb_5000_movies.csv'

def load_data():
    global df, cosine_sim
    
    # A. Check if CSV exists
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            
            # Basic Data Cleaning
            df['overview'] = df['overview'].fillna('')
            
            # Parse Genres (Convert string "[{'name': 'Action'}]" to list "Action")
            # We check if the first row is a string representation of a list
            if isinstance(df['genres'].iloc[0], str) and '[' in df['genres'].iloc[0]:
                df['genres'] = df['genres'].apply(literal_eval)
                df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
                df['genres'] = df['genres'].apply(lambda x: ' '.join(x))
            
            # Ensure 'vote_average' exists for sorting best genre movies
            if 'vote_average' not in df.columns:
                df['vote_average'] = 0

            # Create Content Soup for AI
            df['content'] = df['overview'] + " " + df['genres']
            
            # Train AI Model
            print("⚙️  Training AI model on CSV data...")
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['content'])
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            create_dummy_data()
    else:
        create_dummy_data()

def create_dummy_data():
    global df, cosine_sim
    print("⚠️ CSV not found. Using dummy data for testing.")
    data = {
        'title': ['Avatar', 'Star Wars', 'The Dark Knight', 'Inception', 'Toy Story', 'The Hangover', 'The Exorcist'],
        'genres': ['Action Sci-Fi', 'Action Sci-Fi', 'Action Crime', 'Action Sci-Fi', 'Animation Comedy', 'Comedy', 'Horror'],
        'overview': ['Blue aliens.', 'Space wars.', 'Batman fights Joker.', 'Dream stealing.', 'Toys alive.', 'Bachelor party.', 'Scary possession.'],
        'vote_average': [7.9, 8.2, 8.5, 8.3, 7.9, 7.0, 7.5]
    }
    df = pd.DataFrame(data)
    df['content'] = df['overview'] + " " + df['genres']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Run the load function
load_data()

# ---------------------------------------------------------
# STEP 2: LOGIC FUNCTIONS
# ---------------------------------------------------------

# Logic A: Find similar movies based on a specific movie title
def get_recommendations_by_title(title):
    try:
        # Case-insensitive search for the movie title
        idx_list = df.index[df['title'].str.lower() == title.lower()].tolist()
        
        if not idx_list:
            return []
        
        idx = idx_list[0] # Take the first match

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6] # Top 5 (excluding itself)
        
        movie_indices = [i[0] for i in sim_scores]
        
        results = []
        for i in movie_indices:
            results.append({
                "title": df['title'].iloc[i],
                "genre": df['genres'].iloc[i]
            })
        return results
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return []

# Logic B: Find top movies based on a genre keyword
def get_movies_by_genre(genre_query):
    try:
        # Filter rows where the 'genres' column contains the query string (e.g. "Comedy")
        mask = df['genres'].str.lower().str.contains(genre_query.lower())
        genre_movies = df[mask]
        
        if genre_movies.empty:
            return []

        # Sort by rating (vote_average) so we show GOOD movies, not random ones
        # If 'vote_average' column exists, use it. Otherwise just take head().
        if 'vote_average' in df.columns:
            genre_movies = genre_movies.sort_values(by='vote_average', ascending=False)
        
        # Get top 5
        top_movies = genre_movies.head(5)
        
        results = []
        for index, row in top_movies.iterrows():
            results.append({
                "title": row['title'],
                "genre": row['genres']
            })
        return results

    except Exception as e:
        print(f"Error in genre search: {e}")
        return []

# ---------------------------------------------------------
# STEP 3: API ENDPOINT
# ---------------------------------------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '').strip()
    
    response_text = ""
    movies = []
    
    # 1. Try finding by Title first
    title_recs = get_recommendations_by_title(user_input)
    
    if title_recs:
        response_text = f"Because you liked '{user_input}', you might like:"
        movies = title_recs
    
    else:
        # 2. If no title found, try finding by Genre
        genre_recs = get_movies_by_genre(user_input)
        
        if genre_recs:
            response_text = f"Here are the top rated '{user_input.title()}' movies:"
            movies = genre_recs
        else:
            # 3. Fallback if nothing found
            response_text = "I couldn't find that movie or genre. Try 'Avatar', 'Action', or 'Comedy'!"
            movies = []

    return jsonify({
        "response": response_text,
        "movies": movies
    })

if __name__ == '__main__':
    print("🚀 Server running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)