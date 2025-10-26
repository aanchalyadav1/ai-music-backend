from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, cv2, tensorflow as tf
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import firebase_admin
from firebase_admin import credentials, auth

app = Flask(__name__)
CORS(app)

# Load your trained emotion model
model = tf.keras.models.load_model('emotion_model.keras')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Firebase setup
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred)

# Spotify setup (Replace with your actual keys)
SPOTIFY_CLIENT_ID = "fb79f67e486247c4b8a0308d1c482646"
SPOTIFY_CLIENT_SECRET = "cce4a58dcb5e43b392861a8d20b89a1d"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

@app.route("/detect", methods=["POST"])
def detect_emotion():
    file = request.files.get("image")
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48)) / 255.0
    img = np.expand_dims(img.reshape(48,48,1), axis=0)
    pred = model.predict(img)
    emotion = emotion_labels[np.argmax(pred)]
    return jsonify({"emotion": emotion})

@app.route("/recommend", methods=["POST"])
def recommend_music():
    data = request.get_json()
    emotion = data["emotion"]
    language = data.get("language","english")
    artist = data.get("artist","")
    query = f"{emotion} mood {language} songs {artist}"
    results = sp.search(q=query, type="track", limit=5)
    tracks = [{
        "name": t["name"],
        "artist": t["artists"][0]["name"],
        "preview": t["preview_url"],
        "url": t["external_urls"]["spotify"]
    } for t in results["tracks"]["items"]]
    return jsonify({"songs": tracks})

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    user = auth.create_user(email=data["email"], password=data["password"])
    return jsonify({"message": "Signup success", "uid": user.uid})

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json()
    link = auth.generate_password_reset_link(data["email"])
    return jsonify({"message": "Reset link sent", "link": link})

if __name__ == "__main__":
    app.run(debug=True)
