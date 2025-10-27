# ==========================
# app.py — Flask Backend (Final)
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
from dotenv import load_dotenv

# -----------------------------
# 1️⃣ App Configuration
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow frontend access

# -----------------------------
# 2️⃣ Load Environment Variables
# -----------------------------
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# -----------------------------
# 3️⃣ Load Emotion Detection Model
# -----------------------------
MODEL_PATH = "emotion_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -----------------------------
# 4️⃣ Firebase Initialization
# -----------------------------
cred = credentials.Certificate("firebaseConfig.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# 5️⃣ Spotify API Configuration
# -----------------------------
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise Exception("⚠️ Spotify credentials missing in .env file!")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# -----------------------------
# 6️⃣ Routes
# -----------------------------

@app.route("/")
def home():
    return jsonify({"message": "🎶 AI Music Recommendation Backend Running!"})

# 🎭 Emotion Detection
@app.route("/detect", methods=["POST"])
def detect_emotion():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48)) / 255.0
        img = np.expand_dims(img.reshape(48, 48, 1), axis=0)

        pred = model.predict(img)
        emotion = emotion_labels[np.argmax(pred)]

        return jsonify({"success": True, "emotion": emotion})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 🎵 Music Recommendation
@app.route("/recommend", methods=["POST"])
def recommend_music():
    try:
        data = request.get_json()
        emotion = data.get("emotion")
        language = data.get("language", "english")
        artist = data.get("artist", "")

        if not emotion:
            return jsonify({"success": False, "error": "Emotion not provided"}), 400

        query = f"{emotion} mood {language} songs {artist}"
        results = sp.search(q=query, type="track", limit=5)

        tracks = []
        for t in results["tracks"]["items"]:
            tracks.append({
                "name": t["name"],
                "artist": t["artists"][0]["name"],
                "preview": t["preview_url"],
                "url": t["external_urls"]["spotify"],
                "album": t["album"]["images"][0]["url"] if t["album"]["images"] else None
            })

        return jsonify({"success": True, "songs": tracks})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 🔐 User Signup
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        email = data["email"]
        password = data["password"]

        user = auth.create_user(email=email, password=password)
        db.collection("users").document(user.uid).set({
            "email": email,
            "created": firestore.SERVER_TIMESTAMP
        })

        return jsonify({"success": True, "uid": user.uid, "message": "Signup successful"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 🔑 Forgot Password
@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    try:
        data = request.get_json()
        email = data["email"]

        link = auth.generate_password_reset_link(email)
        return jsonify({"success": True, "link": link})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 🎧 Spotify Callback
@app.route("/callback")
def spotify_callback():
    return jsonify({"message": "Spotify callback received"})


# ✅ Health Check
@app.route("/health")
def health():
    return "OK", 200


# -----------------------------
# Run the App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
