import re
import hashlib
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import spacy

# Initialize NLP components
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

DISASTER_KEYWORDS = {
    'flood': {
        'keywords': ['flood', 'flooding', 'inundation', 'water logging', 'heavy rain'],
        'weight': 1.5
    },
    'cyclone': {
        'keywords': ['cyclone', 'hurricane', 'storm', 'typhoon', 'wind', 'gale'],
        'weight': 1.5
    },
    'earthquake': {
        'keywords': ['earthquake', 'tremor', 'seismic', 'richter', 'epicenter'],
        'weight': 1.3
    },
    'drought': {
        'keywords': ['drought', 'water scarcity', 'water shortage', 'dry spell'],
        'weight': 1.2
    },
    'fire': {
        'keywords': ['fire', 'blaze', 'burn', 'flame', 'inferno', 'wildfire'],
        'weight': 1.3
    },
    'landslide': {
        'keywords': ['landslide', 'mudslide', 'rockfall', 'avalanche'],
        'weight': 1.3
    },
    'epidemic': {
        'keywords': ['epidemic', 'pandemic', 'virus', 'disease', 'outbreak'],
        'weight': 1.2
    },
    'industrial': {
        'keywords': ['industrial accident', 'chemical leak', 'explosion', 'gas leak'],
        'weight': 1.2
    },
    'tsunami': {
        'keywords': ['tsunami', 'tidal wave', 'sea surge'],
        'weight': 1.4
    }
}

def preprocess_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    doc = nlp(text.lower())
    cleaned_tokens = []
    
    for token in doc:
        if not token.is_stop and token.text.strip():
            stemmed = stemmer.stem(token.text)
            lemmatized = token.lemma_
            cleaned_tokens.append(lemmatized)
    
    return ' '.join(cleaned_tokens)

def categorize_disaster(text):
    """Categorize disaster type"""
    text_lower = text.lower()
    
    # Keyword matching first
    disaster_scores = {}
    for disaster_type, data in DISASTER_KEYWORDS.items():
        score = 0
        for keyword in data['keywords']:
            if keyword in text_lower:
                score += data['weight']
        disaster_scores[disaster_type] = score
    
    max_score = max(disaster_scores.values())
    if max_score >= 2.0:
        return max(disaster_scores.items(), key=lambda x: x[1])[0]
    
    # Zero-shot classification as fallback
    try:
        from transformers import pipeline
        classifier = pipeline("zero-shot-classification", 
                            model="facebook/bart-large-mnli")
        labels = list(DISASTER_KEYWORDS.keys()) + ["other"]
        result = classifier(text, labels)
        return result['labels'][0] if result['scores'][0] > 0.5 else "other"
    except:
        return "other"

def calculate_risk_score(probability, locations, disaster_type):
    """Calculate comprehensive risk score"""
    # Base score from probability
    if probability >= 0.85: base = 5
    elif probability >= 0.7: base = 4
    elif probability >= 0.55: base = 3
    elif probability >= 0.4: base = 2
    elif probability >= 0.25: base = 1
    else: base = 0
    
    # Location modifiers
    location_mod = 0
    if locations and locations[0]['name'] != "Unknown":
        loc_name = locations[0]['name']
        
        # Tamil Nadu bonus
        if any(loc.lower() in loc_name.lower() for loc in TAMIL_NADU_LOCATIONS):
            location_mod += 0.5
        
        # Historical disaster bonus
        for area, disasters in HISTORICAL_DISASTER_AREAS.items():
            if area.lower() in loc_name.lower():
                if disaster_type in disasters:
                    location_mod += disasters[disaster_type] * 1.5
                break
    
    return min(5, max(0, round(base + location_mod, 1)))

def get_risk_category(score):
    """Convert score to risk category"""
    if score >= 4.5: return "Very High"
    elif score >= 3.5: return "High"
    elif score >= 2.5: return "Medium-High"
    elif score >= 1.5: return "Medium"
    elif score >= 0.5: return "Low"
    else: return "Very Low"
