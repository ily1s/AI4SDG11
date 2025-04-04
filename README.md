# AI4SDG: Real-Time Disaster Detection from Twitter using BERT

🌍 An AI-powered system to monitor and classify real-time Twitter data for early detection of disasters and urban emergencies, in support of SDG 11 (Sustainable Cities & Communities).

---

## 🔍 Introduction
Urban disasters such as floods, earthquakes, fires, and infrastructure failures can result in massive human and economic losses. Conventional alert systems often suffer from delayed reporting. This project introduces an AI-based solution that leverages BERT-based NLP models to detect disaster events from social media, particularly Twitter, in real time. 

The system is designed to:
- Detect disaster-related tweets
- Classify disaster-related tweets and determine disaster type using a fine-tuned BERT model.
- Extract key details such as location using NER.
- Trigger alerts when disaster activity spikes.

---

## 📊 Data
- Source: Twitter Streaming API using relevant keywords ("flood", "earthquake", "fire", "collapse", etc.)
- Preprocessed to remove retweets, non-English tweets, and spam
- Database: SQLite storage for tweet metadata

---

## 🤖 Models
1️⃣ Disaster Classification Model
	-	Model: Fine-tuned BERT (based on bert-base-uncased)
	-	Task: Predicts whether a tweet describes a real disaster
  -	Disaster type (earthquake, flood, bombing, etc.)
  -	Training Data: Labeled disaster tweets dataset

2️⃣ Named Entity Recognition (NER) for Location & Disaster Type
	-	Model: dslim/bert-base-NER
	-	Task: Extracts:
	-	Locations (cities, countries)
	
Additional Features:
	•	Time-series monitoring for detecting disaster trends
	•	Optional: distilBERT for faster inference

---

⚡ System Architecture

1- Data Collection – Fetches tweets using Twitter API
2- Preprocessing – Cleans and prepares tweet text
3- Classification – Uses fine-tuned BERT to detect disasters & disaster types
4- NER Extraction – Identifies locations 
5- Database Storage – Saves classified tweets & extracted info
6- Alert System – Triggers alerts based on activity spikes

⸻

📊 Dashboard
	•	Real-time disaster tweet monitoring
	•	Geolocation visualization (map-based)
	•	Time-series event detection
	•	Admin panel for analysis & reporting


---

## ✅ Results
- Achieved classification accuracy > 90% on validation dataset
- Successful detection of recent disaster events in test runs
- Alert system triggered based on tweet volume spikes and model confidence

---

## 🚀 Get Started
1. Clone the repo
```bash
git clone https://github.com/ily1s/AI4SDG11.git
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Set up Twitter API keys
	•	Add credentials in .env file
	•	Ensure SQLite database (tweets.db) is initialized
4. Run the main pipeline
```bash
python app.py
```

---

## 📎 Resources
- BERT Disaster Classifier: https://huggingface.co/elam2909/bert-disaster-classifier
- Hugging Face NER Model: https://huggingface.co/dslim/bert-base-NER
- Kaggle Disaster Tweets Dataset: https://www.kaggle.com/datasets/ily4ho/disaster-tweets

---

## 🤝 Contributing
Feel free to submit issues, feature requests, or pull requests. Let’s build tech that saves lives!

---

Built with ❤️ by Team BoTs for GITEX 2025
