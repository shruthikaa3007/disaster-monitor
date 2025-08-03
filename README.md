
# 🌍 Disaster Alert Fusion Plus

**Multi-Source Deep Learning Framework for Disaster Monitoring and Impact Assessment**

Disaster Alert Fusion Plus is an AI-powered disaster monitoring system that integrates real-time news alerts, weather data, satellite imagery, and social media input to detect, assess, and report disaster events. It is particularly designed to support monitoring in regions like Tamil Nadu, India.

---

## 🚀 Features

- 📰 **News Monitoring**
  - Fetches multilingual disaster-related news from RSS feeds
  - Translates non-English news content
  - Detects disaster-related content using BERT and keyword scoring
  - Named Entity Recognition (NER) for extracting locations
  - Duplicate article filtering using TF-IDF cosine similarity

- 🌦️ **Weather Alert Integration**
  - Fetches weather alerts from OpenWeatherMap, IMD, or NOAA
  - Flags alerts like floods, cyclones, and heatwaves

- 🛰️ **Satellite Imagery & Damage Detection**
  - Retrieves pre- and post-disaster images via Google Earth Engine
  - Uses YOLOv8/xView2 for detecting building damage
  - Filters output to highlight "major-damage" or "destroyed" buildings
  - Images processed and stored in Google Cloud Storage

- 📊 **Interactive Dashboard**
  - Built with Streamlit
  - View disaster alerts and risk assessments
  - Filter by location, severity, and event type
  - Download reports as CSV
  - Optionally log results to Firebase

---

## 📂 Project Structure

```
DisasterAlertFusionPlus/
│
├── rss_bert.py               # Location and disaster classification
├── weather_alerts.py         # Weather API integration
├── earth_engine_fetch.py     # Satellite image fetching (GEE)
├── yolo_inference.py         # Damage detection model interface
├── dashboard_app.py          # Streamlit dashboard frontend
├── requirements.txt          # Python dependencies
├── models/                   # Pre-trained ML/DL model weights
├── data/                     # Output logs and sample images
├── utils/                    # Helper functions (translation, scoring, etc.)
└── README.md
```

---

## ⚙️ Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/disaster-alert-fusion-plus.git
cd disaster-alert-fusion-plus
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Up Credentials**
- Google Earth Engine (requires authentication)
- Google Cloud credentials for accessing Storage buckets
- API keys for:
  - OpenWeatherMap or IMD
  - Firebase (optional)

4. **Run the Streamlit App**
```bash
streamlit run dashboard_app.py
```

---

## 🧠 ML Models Used

- **BERT**: For disaster keyword classification and location extraction
- **YOLOv8 / xView2**: For satellite damage detection from GEE imagery
- **TF-IDF + Cosine Similarity**: For duplicate news filtering
- **Ensemble Classifier**: Combines keyword scoring, RF, and BERT prediction scores

---

## 🧪 Example Use Case

**Event**: Chennai Floods (Nov 2022)  
**Pipeline**:
1. News article detected with flooding keywords and Chennai-based locations
2. Satellite images from before and after the event were fetched
3. YOLOv8 inferred significant building damage in Tambaram, T. Nagar, and Saidapet
4. Risk score marked as **High** due to:
   - Verified location
   - Multiple sources reporting same event
   - Detected physical damage via satellite

---

## 🔮 Future Enhancements

- Real-time Twitter stream ingestion
- Mobile version of the dashboard
- Email/SMS-based emergency alert system
- Historical disaster trend visualization



## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
