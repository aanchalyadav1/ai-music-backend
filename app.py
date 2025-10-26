# ==========================
# app.py — Flask Backend
# ==========================
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os

app = Flask(__name__)
CORS(app)  # allows connection from frontend

# -----------------------------
# 1️⃣ Load Emotion Detection Model
# -----------------------------
MODEL_PATH = "emotion_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -----------------------------
# 2️⃣ Initialize Firebase
# -----------------------------
cred = credentials.Certificate("firebaseConfig.json")
firebase_admin.initialize_app(cred)
db = firestore.client()  # Firestore database connection

# -----------------------------
# 3️⃣ Spotify API Setup
# -----------------------------
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# -----------------------------
# 4️⃣ Routes
# -----------------------------

@app.route("/")
def home():
    return jsonify({"message": "AI Music Recommendation Backend is Running 🎵"})


# 🎭 Detect Emotion Route
@app.route("/detect", methods=["POST"])
def detect_emotion():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        # Convert image to grayscale for prediction
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48)) / 255.0
        img = np.expand_dims(img.reshape(48, 48, 1), axis=0)

        pred = model.predict(img)
        emotion = emotion_labels[np.argmax(pred)]

        return jsonify({"emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🎵 Recommend Music Route
@app.route("/recommend", methods=["POST"])
def recommend_music():
    try:
        data = request.get_json()
        emotion = data.get("emotion")
        language = data.get("language", "english")
        artist = data.get("artist", "")

        if not emotion:
            return jsonify({"error": "Emotion not provided"}), 400

        query = f"{emotion} mood {language} songs {artist}"
        results = sp.search(q=query, type="track", limit=5)

        tracks = []
        for t in results["tracks"]["items"]:
            tracks.append({
                "name": t["name"],
                "artist": t["artists"][0]["name"],
                "preview": t["preview_url"],
                "url": t["external_urls"]["spotify"]
            })

        return jsonify({"songs": tracks})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔐 Signup Route (Firebase Auth + Firestore Save)
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        email = data["email"]
        password = data["password"]

        user = auth.create_user(email=email, password=password)

        # Save user info to Firestore
        db.collection("users").document(user.uid).set({
            "email": email,
            "created": firestore.SERVER_TIMESTAMP
        })

        return jsonify({"message": "Signup success", "uid": user.uid})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔑 Forgot Password Route
@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    try:
        data = request.get_json()
        email = data["email"]

        link = auth.generate_password_reset_link(email)
        return jsonify({"message": "Password reset link generated", "link": link})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Health Check (for Render)
@app.route("/health")
def health():
    return "OK", 200


import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

