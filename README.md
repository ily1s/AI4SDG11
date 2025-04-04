# AI4SDG: Real-Time Disaster Detection from Twitter using BERT

🌍 An AI-powered system to monitor and classify real-time Twitter data for early detection of disasters and urban emergencies, in support of SDG 11 (Sustainable Cities & Communities).

---

## 🔍 Introduction
Urban disasters such as floods, earthquakes, fires, and infrastructure failures can result in massive human and economic losses. Conventional alert systems often suffer from delayed reporting. This project introduces an AI-based solution that leverages BERT-based NLP models to detect disaster events from social media, particularly Twitter, in real time. 

The system is designed to:
- Detect disaster-related tweets
- Classify them using a fine-tuned BERT model
- Extract named entities and geolocation info
- Trigger alerts when certain thresholds are met

---

## 📊 Data
- Source: Twitter Streaming API using relevant keywords ("flood", "earthquake", "fire", "collapse", etc.)
- Preprocessed to remove retweets, non-English tweets, and spam
- Annotated disaster tweet dataset from Kaggle (used for fine-tuning)

---

## 🤖 Models
- BERT base model (transformers library by Hugging Face)
- Fine-tuned on disaster classification using tweet data
- Named Entity Recognition (NER) using pretrained model: `dslim/bert-base-NER`

Additional tools:
- Time-series tweet count analysis for event spike detection
- Optional: distilBERT for faster inference in real-time applications

---

## 💡 Applications
- Real-time urban disaster alerting and crisis response
- Smart city emergency infrastructure
- Data source for NGOs, rescue services, and government bodies
- Integration with mobile alerts and dashboards

---

## ✅ Results
- Achieved classification accuracy > 90% on validation dataset
- Successful detection of recent disaster events in test runs
- Alert system triggered based on tweet volume spikes and model confidence

---

## 🚀 Get Started
1. Clone the repo
```bash
git clone https://github.com/your-username/AI4SDG-BERT.git
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Add your Twitter API credentials in config.py
4. Run the main pipeline
```bash
python app.py
```

---

## 📎 Resources
- BERT Disaster Classifier: https://huggingface.co/elam2909/bert-disaster-classifier
- Kaggle Disaster Tweets Dataset: https://www.kaggle.com/c/nlp-getting-started
- Hugging Face NER Model: https://huggingface.co/dslim/bert-base-NER

---

## 🤝 Contributing
Feel free to submit issues, feature requests, or pull requests. Let’s build tech that saves lives!

---

Built with ❤️ by Team BoTs for GITEX 2025