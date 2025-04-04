import torch
import sqlite3
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import joblib
import re

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert_disaster_model")

# Load the model
model = BertForSequenceClassification.from_pretrained("bert_disaster_model")
model.eval()

# Load the vectorizer and classifier for disaster type
vectorizer = joblib.load("models/disaster_tfidf_vectorizer.joblib")
classifier = joblib.load("models/disaster_type_classifier.joblib")

# Load Named Entity Recognition (NER) model for location extraction
ner = pipeline("ner", model="dslim/bert-base-NER")
print("✅ Loaded NER model for location extraction")


def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  # Remove special characters
    return text.strip()


def predict_disaster(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "Disaster" if torch.argmax(probs) == 1 else "Not Disaster"


def predict_disaster_type(text):
    text_vectorized = vectorizer.transform([text])
    return classifier.predict(text_vectorized)[0]


def extract_locations(tweet):
    """Extract location names from a tweet using NER."""
    entities = ner(tweet)
    locations = [
        entity["word"] for entity in entities if entity["entity"] in ["B-LOC", "I-LOC"]
    ]
    return ", ".join(set(locations)) if locations else None  # Convert to string


def classify_and_update_tweets():
    """Classifies tweets, extracts disaster types, locations, and updates the database."""
    with sqlite3.connect("tweets.db") as conn:
        cursor = conn.cursor()

        # Fetch tweets that have not been classified
        cursor.execute(
            "SELECT tweet_id, tweet_text FROM disaster_tweets WHERE disaster_type IS NULL"
        )
        tweets = cursor.fetchall()

        for tweet_id, tweet_text in tweets:
            # Preprocess the tweet text
            tweet_text = preprocess_tweet(tweet_text)
            disaster_status = predict_disaster(tweet_text)

            if disaster_status == "Disaster":
                disaster_type = predict_disaster_type(tweet_text)
                location = extract_locations(tweet_text)
                print(
                    f"✅ Tweet classified as disaster: {disaster_type}, Location: {location}"
                )
            else:
                disaster_type = None
                location = None
                print(f"✅ Tweet classified as not a disaster.")

            # Update database with disaster type and location
            cursor.execute(
                "UPDATE disaster_tweets SET disaster_type = ?, location = ? WHERE tweet_id = ?",
                (disaster_type, location, tweet_id),
            )

        conn.commit()

    print("✅ Tweets classified and updated in the database!")


# Run classification and database update
# classify_and_update_tweets()


tweet = "Bombing in gaza"
disaster_status = predict_disaster(tweet)

if disaster_status == "Disaster":
    disaster_type = predict_disaster_type(tweet)
    print(f"The tweet is a disaster and belongs to type: {disaster_type}")
else:
    print("The tweet is not about a disaster.")
