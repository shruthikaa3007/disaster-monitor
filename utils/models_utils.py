import os
import joblib
import numpy as np
import torch
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

def download_model(bucket_name, source_path, local_path):
    """Download model from GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(local_path)
    print(f"Downloaded {source_path} to {local_path}")

def load_models():
    """Load all required models"""
    models = {}
    
    # 1. Load Random Forest model
    local_rf_path = '/tmp/rf_model.joblib'
    if not os.path.exists(local_rf_path):
        download_model('disaster-monitor-models', 'rf_model.joblib', local_rf_path)
    models['rf_model'] = joblib.load(local_rf_path)
    
    # 2. Load BERT tokenizer and model
    models['tokenizer'] = AutoTokenizer.from_pretrained("bert-base-uncased")
    models['base_model'] = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # 3. Load Sentence Transformer
    models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
    
    return models

def predict_disaster(text, models):
    """Run ensemble prediction"""
    # Feature extraction
    features = extract_features(text, models['sentence_model'])
    
    # RF prediction
    rf_features = np.hstack((
        features["embedding"],
        list(features["keyword_scores"].values()),
        [sum(features["entity_counts"].values()), features["tn_location_present"]]
    ))
    rf_prob = models['rf_model'].predict_proba([rf_features])[0][1]
    
    # Transformer prediction
    inputs = models['tokenizer'](text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = models['base_model'](**inputs)
    transformer_probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()
    transformer_prob = transformer_probs[1]
    
    # Ensemble probability
    ensemble_prob = (0.5 * rf_prob) + (0.3 * transformer_prob) + (0.2 * features['keyword_score_total']/10.0)
    
    return {
        "probability": float(ensemble_prob),
        "features": features
    }

def extract_features(text, sentence_model):
    """Extract features from text"""
    # Sentence embedding
    embedding = sentence_model.encode(text)
    
    # Keyword scores
    keyword_scores = {}
    keyword_score_total = 0
    text_lower = text.lower()
    
    for disaster_type, data in DISASTER_KEYWORDS.items():
        score = 0
        for keyword in data['keywords']:
            if keyword in text_lower:
                score += data['weight']
        keyword_scores[f"keyword_{disaster_type}"] = score
        keyword_score_total += score
    
    # Entity counts
    doc = nlp(text)
    entity_counts = {}
    for ent in doc.ents:
        entity_type = ent.label_
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    # Tamil Nadu location check
    tn_location_present = 0
    for location in TAMIL_NADU_LOCATIONS:
        if location.lower() in text_lower:
            tn_location_present = 1
            break
    
    return {
        "embedding": embedding,
        "keyword_scores": keyword_scores,
        "keyword_score_total": keyword_score_total,
        "entity_counts": entity_counts,
        "tn_location_present": tn_location_present
    }
