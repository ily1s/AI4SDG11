from transformers import pipeline
import joblib
from huggingface_hub import hf_hub_download

# This is our fine-tuned BERT model for disaster classification.
# You can find the model on Hugging Face: https://huggingface.co/elam2909/bert-disaster-classifier

classifier = pipeline("text-classification", model="elam2909/bert-disaster-classifier")

# Download the joblib files from Hugging Face
tfidf_path = hf_hub_download(
    repo_id="elam2909/bert-disaster-classifier",
    filename="disaster_tfidf_vectorizer.joblib",
)

type_classifier_path = hf_hub_download(
    repo_id="elam2909/bert-disaster-classifier",
    filename="disaster_type_classifier.joblib",
)

# Load the models
tfidf_vectorizer = joblib.load(tfidf_path)
disaster_type_classifier = joblib.load(type_classifier_path)

# Example text
text = "Huge earthquake just hit southern Turkey. Buildings are collapsing!"

result = classifier(text)
print(result)

# The output will contain the classification and confidence score
# Check if it's classified as a disaster
is_disaster = result[0]["label"] == "disaster"


# If it's a disaster, classify the type
if is_disaster:
    # Transform the text using the TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([text])

    # Predict the disaster type
    disaster_type = disaster_type_classifier.predict(text_vectorized)[0]

    print(f"Disaster type: {disaster_type}")

    # If you want to see the probability distribution across disaster types
    disaster_type_proba = disaster_type_classifier.predict_proba(text_vectorized)[0]
    disaster_types = disaster_type_classifier.classes_

    # Print probabilities for each disaster type
    for dtype, prob in zip(disaster_types, disaster_type_proba):
        print(f"{dtype}: {prob:.4f}")
else:
    print("Text classified as not being about a disaster.")
