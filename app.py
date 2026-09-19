import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load the final website models once when the server starts.
hausa_model = joblib.load(MODEL_DIR / "hausa_model.pkl")
hausa_tfidf = joblib.load(MODEL_DIR / "hausa_tfidf.pkl")
yoruba_model = joblib.load(MODEL_DIR / "yoruba_model.pkl")
yoruba_tfidf = joblib.load(MODEL_DIR / "yoruba_tfidf.pkl")

# Lightweight vocabulary-based validation. This is a validation layer,
# not a separately trained language-identification model.
HAUSA_WORDS = {
    "da", "shi", "ita", "su", "ne", "ce", "wannan", "haka", "amma",
    "yanzu", "gobe", "yau", "jami", "gwamnati", "shugaban", "kasar",
    "mutane", "labarai", "kuma", "saboda", "idan", "wanda", "waɗanda",
    "akan", "cikin", "domin", "zuwa", "daga", "tare", "bisa", "wata",
    "wani", "wannan", "an", "ya", "ta", "sun", "za", "ba", "babu",
    "makaranta", "dalibai", "kudi", "noma", "tsaro", "sojoji", "yan sanda"
}
YORUBA_WORDS = {
    "ati", "ni", "o", "ni", "mo", "a", "won", "wa", "ti", "ko", "fun",
    "lati", "pẹlu", "pelu", "ṣugbọn", "sugbon", "bayi", "ọjọ", "ojo",
    "ọ̀rọ̀", "oro", "àwọn", "awon", "orilẹ", "orile", "ede", "yìí", "yii",
    "ìròyìn", "iroyin", "ijọba", "ijoba", "ààrẹ", "aare", "orílẹ̀-èdè",
    "ile", "ilu", "eniyan", "awon", "ti", "se", "jẹ", "je", "kan", "e",
    "wọn", "won", "rẹ", "re", "mi", "wa", "yin", "fun", "ninu", "lori"
}

def language_score(text, vocabulary):
    words = {w.strip(".,!?;:()[]{}\"'“”‘’").lower() for w in text.split()}
    words.discard("")
    if not words:
        return 0.0
    return sum(word in vocabulary for word in words) / len(words)


def validate_language(text, selected):
    """Return (ok, detected, score). Short text is treated cautiously."""
    hausa_score = language_score(text, HAUSA_WORDS)
    yoruba_score = language_score(text, YORUBA_WORDS)

    if len(text.split()) < 5:
        return False, "unknown", 0.0

    if selected == "hausa":
        return hausa_score >= 0.08 and hausa_score >= yoruba_score * 1.15, "hausa", hausa_score
    return yoruba_score >= 0.08 and yoruba_score >= hausa_score * 1.15, "yoruba", yoruba_score


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    language = str(data.get("language", "")).strip().lower()
    news_text = str(data.get("text", "")).strip()

    if language not in {"hausa", "yoruba"}:
        return jsonify({"error": "Please select Hausa or Yoruba."}), 400
    if not news_text:
        return jsonify({"error": "Please enter news text."}), 400

    ok, detected, language_score_value = validate_language(news_text, language)
    if not ok:
        if detected == "unknown":
            message = "The text is too short or could not be identified. Please enter a longer Hausa or Yoruba news text."
        else:
            message = f"The entered text does not appear to be {language.capitalize()}. Please enter news in {language.capitalize()}."
        return jsonify({
            "error": message,
            "language_valid": False,
            "detected_language": detected
        }), 422

    try:
        if language == "hausa":
            features = hausa_tfidf.transform([news_text])
            prediction = int(hausa_model.predict(features)[0])
            probabilities = hausa_model.predict_proba(features)[0]
        else:
            features = yoruba_tfidf.transform([news_text])
            prediction = int(yoruba_model.predict(features)[0])
            probabilities = yoruba_model.predict_proba(features)[0]

        confidence = float(max(probabilities) * 100)
        result = "Genuine" if prediction == 0 else "Fake/Misleading"

        return jsonify({
            "language": language,
            "prediction": result,
            "confidence": round(confidence, 2),
            "language_valid": True,
            "detected_language": detected
        })
    except Exception:
        return jsonify({"error": "An error occurred while processing the news."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
