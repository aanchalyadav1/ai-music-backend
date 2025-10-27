# ==========================
# app.py — Flask Backend (Final Corrected)
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
CORS(app, resources={r"/*": {"origins": "*"}})

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
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Emotion detection model loaded successfully.")
except Exception as e:
    print("⚠️ Failed to load emotion model:", e)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -----------------------------
# 4️⃣ Firebase Initialization
# -----------------------------
try:
    cred = credentials.Certificate("firebaseConfig.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print("⚠️ Firebase initialization failed:", e)
    db = None

# -----------------------------
# 5️⃣ Spotify Configuration
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
        # Check if image was uploaded
        if "image" not in request.files:
            print("❌ No file found in request.files")
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["image"]
        filename = file.filename or "uploaded_image.jpg"

        # Temporary save (Render allows /tmp/)
        os.makedirs("/tmp/uploads", exist_ok=True)
        temp_path = os.path.join("/tmp/uploads", filename)
        file.save(temp_path)
        print(f"✅ File saved temporarily: {temp_path}")

        # Read and preprocess image
        img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image format"}), 400

        img = cv2.resize(img, (48, 48)) / 255.0
        img = np.expand_dims(img.reshape(48, 48, 1), axis=0)

        pred = model.predict(img)
        emotion = emotion_labels[np.argmax(pred)]

        print(f"🎭 Detected Emotion: {emotion}")
        return jsonify({"success": True, "emotion": emotion})

    except Exception as e:
        print("🔥 Error in /detect:", e)
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
        print("🔥 Error in /recommend:", e)
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
        print("🔥 Error in /signup:", e)
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
        print("🔥 Error in /forgot-password:", e)
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
