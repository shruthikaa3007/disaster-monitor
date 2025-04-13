import time
from geopy.geocoders import Nominatim
from geopy.exceptions import GeocoderTimedOut, GeocoderUnavailable

TAMIL_NADU_LOCATIONS = [
    "Tamil Nadu", "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", 
    "Tiruppur", "Salem", "Erode", "Tirunelveli", "Vellore", "Thoothukudi",
    "Dindigul", "Thanjavur", "Ranipet", "Sivakasi", "Karur", "Udhagamandalam",
    "Hosur", "Nagercoil", "Kanchipuram", "Kumarapalayam", "Karaikkudi",
    "Neyveli", "Cuddalore", "Kumbakonam", "Tiruvannamalai", "Pollachi",
    "Rajapalayam", "Gudiyatham", "Pudukkottai", "Vaniyambadi", "Ambur", 
    "Nagapattinam"
]

HISTORICAL_DISASTER_AREAS = {
    "Chennai": {"flood": 0.8, "cyclone": 0.6},
    "Cuddalore": {"cyclone": 0.9, "flood": 0.7},
    "Nagapattinam": {"tsunami": 0.9, "cyclone": 0.8, "flood": 0.7},
    "Kanyakumari": {"tsunami": 0.8},
    "Nilgiris": {"landslide": 0.8},
    "Coimbatore": {"drought": 0.6},
    "Ramanathapuram": {"drought": 0.7},
    "Thoothukudi": {"cyclone": 0.6, "industrial": 0.7},
}

geolocator = Nominatim(user_agent="disaster-monitor")

def get_coordinates(location_name, max_retries=3):
    """Get coordinates with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(f"{location_name}, Tamil Nadu, India")
            if location:
                return {
                    "name": location_name,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "address": location.address,
                    "confidence": 1.0 - (attempt * 0.2)  # Reduce confidence with retries
                }
            return None
        except (GeocoderTimedOut, GeocoderUnavailable):
            if attempt == max_retries - 1:
                return {
                    "name": location_name,
                    "latitude": None,
                    "longitude": None,
                    "address": None,
                    "confidence": 0.1
                }
            time.sleep(1 * (attempt + 1))

def extract_locations(text, nlp):
    """Extract locations with coordinates from text"""
    doc = nlp(text)
    text_lower = text.lower()
    
    # Find locations with confidence scores
    locations_found = {}
    
    # Exact matches for Tamil Nadu locations
    for loc in TAMIL_NADU_LOCATIONS:
        if loc.lower() in text_lower:
            locations_found[loc] = 1.0  # High confidence for exact matches
    
    # NER locations
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            loc = ent.text
            if loc not in locations_found:
                locations_found[loc] = 0.5  # Lower confidence for NER
    
    # Get coordinates for top locations (sorted by confidence)
    results = []
    for loc, conf in sorted(locations_found.items(), key=lambda x: x[1], reverse=True)[:3]:
        coords = get_coordinates(loc)
        if coords:
            coords["confidence"] = min(coords["confidence"], conf)
            results.append(coords)
    
    return results if results else [{
        "name": "Unknown",
        "latitude": None,
        "longitude": None,
        "address": None,
        "confidence": 0
    }]
