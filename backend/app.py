import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from transformers import pipeline
from river import linear_model, metrics, stream
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymongo
import requests
from bs4 import BeautifulSoup
import pickle

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.getenv("FRONTEND_URL", "*")}})  # Allow frontend URL

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(mongo_uri)
db = client["pitch_deck_db"]
decks_collection = db["decks"]
market_data_collection = db["market_data"]
models_collection = db["models"]

# Initialize NLP model (Hugging Face)
nlp = pipeline("text-classification", model="distilbert-base-uncased")

# Initialize online learning model
def load_model():
    model_data = models_collection.find_one({"name": "online_model"})
    if model_data:
        return pickle.loads(model_data["model"])
    return linear_model.LogisticRegression()

def save_model(model):
    model_data = pickle.dumps(model)
    models_collection.update_one(
        {"name": "online_model"},
        {"$set": {"model": model_data, "updated_at": pd.Timestamp.now()}},
        upsert=True
    )

online_model = load_model()
metric = metrics.Accuracy()

# Preprocessing function
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

# Extract text and images from PDF
def extract_pdf_content(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    images = []
    for page in doc:
        text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append(base_image["image"])
    return text, images

# OCR for images
def ocr_images(images):
    text = ""
    for img_data in images:
        img = Image.open(io.BytesIO(img_data))
        text += pytesseract.image_to_string(img)
    return text

# Evaluate pitch deck
def evaluate_deck(text):
    processed_text = preprocess_text(text)
    clarity_score = nlp(processed_text)[0]['score'] if nlp(processed_text)[0]['label'] == 'POSITIVE' else 1 - nlp(processed_text)[0]['score']
    scores = {
        "problem": clarity_score * 20,
        "solution": clarity_score * 20,
        "market": clarity_score * 20,
        "team": clarity_score * 15,
        "financials": clarity_score * 15,
        "presentation": clarity_score * 10
    }
    total_score = sum(scores.values())
    return scores, total_score

# Scrape market data (Crunchbase API or TechCrunch fallback)
def scrape_market_data():
    api_key = os.getenv("CRUNCHBASE_API_KEY")
    if api_key:
        try:
            url = f"https://api.crunchbase.com/v4/data/searches/organizations"
            headers = {"X-cb-user-key": api_key}
            payload = {
                "field_ids": ["name", "short_description", "funding_total"],
                "query": [{"type": "predicate", "field_id": "funding_total", "operator_id": "gt", "values": [1000000]}],
                "limit": 5
            }
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                market_text = " ".join([org["name"] + ": " + org.get("short_description", "") for org in data.get("entities", [])])
                market_data_collection.insert_one({"text": market_text, "date": pd.Timestamp.now()})
                return market_text
        except Exception as e:
            print(f"Crunchbase API error: {e}")
    # Fallback: Scrape TechCrunch
    try:
        url = "https://techcrunch.com/category/startups/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('h2', class_='post-block__title')[:5]
        market_text = " ".join([article.text.strip() for article in articles])
        market_data_collection.insert_one({"text": market_text, "date": pd.Timestamp.now()})
        return market_text
    except Exception as e:
        print(f"TechCrunch scraping error: {e}")
        return ""

# Online learning update
def update_model(text, feedback=None):
    processed_text = preprocess_text(text)
    X = {"text": processed_text}
    y = feedback if feedback else 1  # Default positive feedback
    online_model.learn_one(X, y)
    metric.update(y, online_model.predict_one(X))
    save_model(online_model)

@app.route('/upload', methods=['POST'])
def upload_deck():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    text, images = extract_pdf_content(file)
    image_text = ocr_images(images)
    full_text = text + " " + image_text
    
    # Evaluate deck
    scores, total_score = evaluate_deck(full_text)
    
    # Store in MongoDB
    deck_data = {
        "text": full_text,
        "scores": scores,
        "total_score": total_score,
        "timestamp": pd.Timestamp.now()
    }
    decks_collection.insert_one(deck_data)
    
    # Update model with new data
    update_model(full_text)
    
    return jsonify({
        "scores": scores,
        "total_score": total_score,
        "feedback": "Evaluation complete. Scores reflect pitch deck quality."
    })

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
