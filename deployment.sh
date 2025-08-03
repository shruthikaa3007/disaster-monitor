#!/bin/bash
# Disaster Assessment System Deployment Script

# Exit on error
set -e

# Variables
PROJECT_ID="disaster-finall"
REGION="us-central1"
STORAGE_BUCKET="$PROJECT_ID-storage"
#NEWS_API_KEY="a5ac08be57fb4059b1ef3c0921cbdca8"  # Replace with your actual key
GEOCODING_API_KEY="AIzaSyDHNrkZCiMbtNq7-aMUgPGJfxUwZrDmxfI"  # Replace with your actual key
ROBOFLOW_API_KEY="ymIHPOetwVTt9MjhUqsC"  # Replace with your actual key
ROBOFLOW_PROJECT="xview2-xbd"
ROBOFLOW_VERSION="8"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Deploying Disaster Assessment System to Google Cloud...${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Create temporary working directory
WORK_DIR=$(mktemp -d)
echo -e "${YELLOW}Working in temporary directory: ${WORK_DIR}${NC}"

# Create project if it doesn't exist
echo -e "${YELLOW}Setting up Google Cloud project...${NC}"
gcloud projects describe $PROJECT_ID &> /dev/null || gcloud projects create $PROJECT_ID --name="Disaster Assessment System"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}Enabling required services...${NC}"
gcloud services enable earthengine.googleapis.com \
    compute.googleapis.com \
    cloudfunctions.googleapis.com \
    pubsub.googleapis.com \
    cloudbuild.googleapis.com \
    cloudscheduler.googleapis.com \
    bigquery.googleapis.com \
    run.googleapis.com \
    cloudresourcemanager.googleapis.com

# Create storage bucket
echo -e "${YELLOW}Creating Cloud Storage bucket...${NC}"
gsutil ls -b gs://$STORAGE_BUCKET &> /dev/null || gsutil mb -l $REGION gs://$STORAGE_BUCKET

# Create BigQuery dataset
echo -e "${YELLOW}Creating BigQuery dataset...${NC}"
#bq --location=$REGION mk --dataset --description "Disaster Assessment Data" $PROJECT_ID:disaster_data

# Create Pub/Sub topics
echo -e "${YELLOW}Creating Pub/Sub topics...${NC}"
topics=("disaster-events" "geocoded-locations" "satellite-images" "damage-assessment-results")
for topic in "${topics[@]}"; do
    gcloud pubsub topics describe $topic &> /dev/null || gcloud pubsub topics create $topic
done

# Set up the News Fetcher function
echo -e "${YELLOW}Setting up News Fetcher function...${NC}"
mkdir -p $WORK_DIR/news-fetcher
cat > $WORK_DIR/news-fetcher/main.py << 'EOF'
# main.py
import json
import os
import requests
from datetime import datetime
from google.cloud import pubsub_v1
import nltk
from nltk.tokenize import sent_tokenize
from google.cloud import bigquery
import pandas as pd

# Download NLTK data
nltk.download('punkt')

# Set up GCP clients
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(os.environ.get('GCP_PROJECT'), 'disaster-events')
bq_client = bigquery.Client()

# List of disaster keywords to search for
DISASTER_KEYWORDS = [
    'earthquake', 'flood', 'hurricane', 'tsunami', 'wildfire', 'tornado',
    'landslide', 'volcanic eruption', 'typhoon', 'cyclone', 'drought'
]

def classify_disaster_type(article_text):
    article_text = article_text.lower()
    for disaster in DISASTER_KEYWORDS:
        if disaster in article_text:
            return disaster
    return None

def extract_locations(article_text):
    locations = []
    sentences = sent_tokenize(article_text)
    location_indicators = ['in', 'at', 'near', 'around']

    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            if word.lower() in location_indicators and i < len(words) - 1:
                candidate = words[i+1]
                if candidate[0].isupper():
                    locations.append(candidate.strip(',.;:()[]{}'))

    return list(set(locations))

def fetch_news():
    """Fetch disaster-related news articles from GDELT."""
    try:
        # Get the latest GDELT update file
        update_response = requests.get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt")
        latest_file = update_response.text.strip().split(' ')[-1].replace('.zip', '.CSV.zip')
        data_url = f"http://data.gdeltproject.org/gdeltv2/{latest_file}"

        # Read GDELT CSV directly from zip
        df = pd.read_csv(data_url, compression='zip', sep='\t', header=None, dtype=str)
        df.columns = [f'col{i}' for i in range(len(df.columns))]

        # GDELT themes and URLs are in columns 57 and 60 (col57 and col60)
        filtered = df[df['col57'].str.contains('|'.join(DISASTER_KEYWORDS), case=False, na=False)]

        articles = []
        for _, row in filtered.iterrows():
            articles.append({
                'title': row['col21'] or '',            # Source URL as title
                'description': row['col23'] or '',      # Extras as description
                'content': row['col23'] or '',
                'url': row['col21'],
                'publishedAt': row['col1']              # Timestamp
            })

        return articles

    except Exception as e:
        print(f"Error fetching GDELT data: {e}")
        return []

def process_articles(articles):
    disaster_events = []
    for article in articles:
        title = article.get('title', '')
        content = article.get('content', '')
        description = article.get('description', '')

        full_text = f"{title} {description} {content}"
        disaster_type = classify_disaster_type(full_text)

        if disaster_type:
            locations = extract_locations(full_text)
            if locations:
                for location in locations:
                    event = {
                        'disaster_type': disaster_type,
                        'location': location,
                        'title': title,
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'timestamp': datetime.now().isoformat()
                    }
                    disaster_events.append(event)
    return disaster_events


# Make sure `save_to_bigquery()` is already defined as in your original script


def save_to_bigquery(events):
    """Save processed events to BigQuery for historical analysis."""
    table_id = f"{os.environ.get('GCP_PROJECT')}.disaster_data.events"
    
    # Ensure the dataset and table exist
    try:
        dataset_ref = bigquery.DatasetReference(os.environ.get('GCP_PROJECT'), 'disaster_data')
        bq_client.get_dataset(dataset_ref)
    except Exception:
        bq_client.create_dataset('disaster_data')
        
    schema = [
        bigquery.SchemaField("disaster_type", "STRING"),
        bigquery.SchemaField("location", "STRING"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("url", "STRING"),
        bigquery.SchemaField("published_at", "STRING"),
        bigquery.SchemaField("timestamp", "TIMESTAMP")
    ]
    
    table = bigquery.Table(table_id, schema=schema)
    try:
        bq_client.get_table(table)
    except Exception:
        bq_client.create_table(table)
    
    rows_to_insert = [
        {
            "disaster_type": event["disaster_type"],
            "location": event["location"],
            "title": event["title"],
            "url": event["url"],
            "published_at": event["published_at"],
            "timestamp": event["timestamp"]
        }
        for event in events
    ]
    
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        print(f"Errors inserting rows: {errors}")
    else:
        print(f"Inserted {len(rows_to_insert)} rows to BigQuery")

def disaster_news_handler(request):
    try:
        articles = fetch_news()
        events = process_articles(articles)

        # Save to BigQuery (reuse your function if defined elsewhere)
        save_to_bigquery(events)

        # Publish to Pub/Sub
        for event in events:
            data = json.dumps(event).encode('utf-8')
            publisher.publish(topic_path, data=data)

        return f"Processed {len(articles)} articles, found {len(events)} disaster events", 200

    except Exception as e:
        print(f"Error processing news: {e}")
        return f"Error: {str(e)}", 500

EOF

cat > $WORK_DIR/news-fetcher/requirements.txt << 'EOF'
google-cloud-pubsub  # Compatible with Python 3.10
nltk
requests
numpy
pandas
pyarrow
google-cloud-bigquery # Newer version that works with Python 3.10


EOF

echo -e "${YELLOW}Setting up Cloud Function deployment...${NC}"

# Check if the Cloud Function exists

gcloud functions deploy news-fetcher \
  --region=us-central1 \
  --runtime=python310 \
  --source=$WORK_DIR/news-fetcher \
  --entry-point=disaster_news_handler \
  --trigger-http \
  --timeout=540s \
  --memory=512MB \
  --set-env-vars=GCP_PROJECT=disaster-finall



# Set up a Cloud Scheduler job to trigger the news fetcher
echo -e "${YELLOW}Setting up Cloud Scheduler job...${NC}"

# Check if the job exists
gcloud scheduler jobs describe news-fetcher-job --location=us-central1 > /dev/null 2>&1 || \
  gcloud scheduler jobs create http news-fetcher-job \
    --location=us-central1 \
    --schedule="0 */2 * * *" \
    --uri="https://us-central1-disaster-finall.cloudfunctions.net/news-fetcher" \
    --http-method=GET \
    --oidc-service-account-email=9042995723-compute@developer.gserviceaccount.com


# Set up the Location Geocoder function
echo -e "${YELLOW}Setting up Location Geocoder function...${NC}"
mkdir -p $WORK_DIR/geocoder
cat > $WORK_DIR/geocoder/main.py << 'EOF'
# Insert main.py content for geocoder here
# main.py
import base64
import json
import os
import requests
from google.cloud import pubsub_v1
from google.cloud import bigquery

# Set up GCP clients
publisher = pubsub_v1.PublisherClient()
output_topic_path = publisher.topic_path(os.environ.get('GCP_PROJECT'), 'geocoded-locations')
bq_client = bigquery.Client()

# Geocoding API
GEOCODING_API_KEY = os.environ.get('GEOCODING_API_KEY')
GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"

def geocode_location(location_name):
    """Convert location name to geographic coordinates and boundary."""
    params = {
        'address': location_name,
        'key': GEOCODING_API_KEY
    }
    
    response = requests.get(GEOCODING_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        if results['status'] == 'OK' and results['results']:
            result = results['results'][0]
            
            # Extract coordinates
            location = result['geometry']['location']
            lat, lng = location['lat'], location['lng']
            
            # Extract administrative area if available
            admin_area = ''
            for component in result['address_components']:
                if 'administrative_area_level_1' in component['types']:
                    admin_area = component['long_name']
                    break
            
            # Extract viewport for approximate boundary
            viewport = result['geometry'].get('viewport', {})
            
            return {
                'name': location_name,
                'formatted_address': result['formatted_address'],
                'latitude': lat,
                'longitude': lng,
                'admin_area': admin_area,
                'viewport': viewport
            }
    
    return None

def save_to_bigquery(geocoded_data, disaster_event):
    """Save geocoded data to BigQuery for historical analysis."""
    table_id = f"{os.environ.get('GCP_PROJECT')}.disaster_data.geocoded_locations"
    
    # Ensure the dataset and table exist
    try:
        dataset_ref = bigquery.DatasetReference(os.environ.get('GCP_PROJECT'), 'disaster_data')
        bq_client.get_dataset(dataset_ref)
    except Exception:
        bq_client.create_dataset('disaster_data')
        
    schema = [
        bigquery.SchemaField("location_name", "STRING"),
        bigquery.SchemaField("formatted_address", "STRING"),
        bigquery.SchemaField("latitude", "FLOAT"),
        bigquery.SchemaField("longitude", "FLOAT"),
        bigquery.SchemaField("admin_area", "STRING"),
        bigquery.SchemaField("disaster_type", "STRING"),
        bigquery.SchemaField("news_title", "STRING"),
        bigquery.SchemaField("news_url", "STRING"),
        bigquery.SchemaField("viewport", "STRING")
    ]
    
    table = bigquery.Table(table_id, schema=schema)
    try:
        bq_client.get_table(table)
    except Exception:
        bq_client.create_table(table)
    
    row = {
        "location_name": geocoded_data["name"],
        "formatted_address": geocoded_data["formatted_address"],
        "latitude": geocoded_data["latitude"],
        "longitude": geocoded_data["longitude"],
        "admin_area": geocoded_data["admin_area"],
        "disaster_type": disaster_event["disaster_type"],
        "news_title": disaster_event["title"],
        "news_url": disaster_event["url"],
        "viewport": json.dumps(geocoded_data["viewport"])
    }
    
    errors = bq_client.insert_rows_json(table_id, [row])
    if errors:
        print(f"Errors inserting rows: {errors}")
    else:
        print(f"Inserted geocoded data to BigQuery")

def geocode_location_handler(event, context):
    """Cloud Function entry point for geocoding locations."""
    try:
        # Parse the Pub/Sub message
        pubsub_message = base64.b64decode(event['data']).decode('utf-8')
        disaster_event = json.loads(pubsub_message)
        
        # Extract location to geocode
        location_name = disaster_event.get('location')
        
        if not location_name:
            print("No location found in disaster event")
            return
        
        # Geocode the location
        geocoded_data = geocode_location(location_name)
        
        if geocoded_data:
            # Save to BigQuery
            save_to_bigquery(geocoded_data, disaster_event)
            
            # Add geocoding data to the disaster event
            enriched_event = {**disaster_event, 'geocoded_data': geocoded_data}
            
            # Publish to next topic for Earth Engine processing
            output_data = json.dumps(enriched_event).encode('utf-8')
            publisher.publish(output_topic_path, data=output_data)
            
            print(f"Successfully geocoded location: {location_name}")
        else:
            print(f"Could not geocode location: {location_name}")
        
    except Exception as e:
        print(f"Error in geocoding function: {e}")
EOF

cat > $WORK_DIR/geocoder/requirements.txt << 'EOF'
google-cloud-pubsub
google-cloud-bigquery
requests
numpy
pyarrow

EOF

echo -e "${YELLOW}Setting up Location Geocoder Cloud Function...${NC}"

# Check if the Cloud Function exists
gcloud functions describe location-geocoder --region=$REGION > /dev/null 2>&1 || \
  gcloud functions deploy location-geocoder \
    --region=$REGION \
    --runtime=python310 \
    --source=$WORK_DIR/geocoder \
    --entry-point=geocode_location_handler \
    --trigger-topic=disaster-events \
    --timeout=540s \
    --memory=512MB \
    --set-env-vars=GCP_PROJECT=$PROJECT_ID,GEOCODING_API_KEY=$GEOCODING_API_KEY


# Set up the Earth Engine function
echo -e "${YELLOW}Setting up Earth Engine function...${NC}"
mkdir -p $WORK_DIR/earth-engine
cat > $WORK_DIR/earth-engine/main.py << 'EOF'
# Insert main.py content for earth-engine here
# main.py
import base64
import json
import os
import ee
from datetime import datetime, timedelta
from google.cloud import storage
from google.cloud import pubsub_v1

# Initialize Earth Engine
ee.Initialize(project='disaster-finall')

# Set up GCP clients
publisher = pubsub_v1.PublisherClient()
output_topic_path = publisher.topic_path(os.environ.get('GCP_PROJECT'), 'satellite-images')
storage_client = storage.Client()

# Set up Cloud Storage bucket for images
BUCKET_NAME = os.environ.get('STORAGE_BUCKET', 'disaster-assessment-images')

def get_bounding_box(viewport):
    """Convert viewport to Earth Engine bounding box."""
    try:
        # Extract coordinates from viewport
        southwest = viewport['southwest']
        northeast = viewport['northeast']
        
        # Create coordinates list
        coords = [
            [southwest['lng'], southwest['lat']],  # SW
            [northeast['lng'], southwest['lat']],  # SE
            [northeast['lng'], northeast['lat']],  # NE
            [southwest['lng'], northeast['lat']],  # NW
            [southwest['lng'], southwest['lat']]   # SW again to close the polygon
        ]
        
        # Create Earth Engine geometry
        return ee.Geometry.Polygon([coords])
    except Exception as e:
        print(f"Error creating bounding box: {e}")
        # Fallback: create small box around the point
        lat = (viewport['southwest']['lat'] + viewport['northeast']['lat']) / 2
        lng = (viewport['southwest']['lng'] + viewport['northeast']['lng']) / 2
        return ee.Geometry.Point([lng, lat]).buffer(10000)  # 10km buffer

def get_satellite_images(region, disaster_type, date_range=7):
    """Get satellite imagery for the given region."""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_range)
        
        # Format dates for Earth Engine
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Different collection strategies based on disaster type
        if disaster_type in ['flood', 'tsunami']:
            # For floods, use Sentinel-1 (SAR) which can see through clouds
            collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(region) \
                .filterDate(start_date_str, end_date_str) \
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
            
            if collection.size().getInfo() > 0:
                image = collection.mosaic()
                visualized = image.select(['VV']).clip(region)
                return visualized, 'COPERNICUS/S1_GRD'
        
        # Default to Sentinel-2 for most disaster types
        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(region) \
            .filterDate(start_date_str, end_date_str) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
        if collection.size().getInfo() > 0:
            image = collection.median()
            visualized = image.select(['B4', 'B3', 'B2']).clip(region)
            return visualized, 'COPERNICUS/S2_SR'
        
        # Fallback to Landsat if Sentinel-2 doesn't have coverage
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(region) \
            .filterDate(start_date_str, end_date_str)
        
        if collection.size().getInfo() > 0:
            image = collection.median()
            visualized = image.select(['SR_B4', 'SR_B3', 'SR_B2']).clip(region)
            return visualized, 'LANDSAT/LC08/C02/T1_L2'
        
        return None, None
        
    except Exception as e:
        print(f"Error getting satellite images: {e}")
        return None, None

def export_images_to_storage(image, location_name, disaster_type, image_source):
    """Export images to Cloud Storage for further processing."""
    try:
        # Create safe filename
        safe_name = ''.join(c if c.isalnum() else '_' for c in location_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_name}_{disaster_type}_{timestamp}"
        
        # Get the bucket
        try:
            bucket = storage_client.get_bucket(BUCKET_NAME)
        except Exception:
            bucket = storage_client.create_bucket(BUCKET_NAME)
        
        # Export parameters for RGB visualization
        rgb_export = {
            'image': image,
            'description': f'{filename}_rgb',
            'bucket': BUCKET_NAME,
            'fileNamePrefix': f'images/{filename}_rgb',
            'scale': 10,
            'maxPixels': 1e9
        }
        
        # Start export task
        task = ee.batch.Export.image.toCloudStorage(**rgb_export)
        task.start()
        
        # Return metadata about the export
        return {
            'filename': filename,
            'bucket': BUCKET_NAME,
            'path': f'images/{filename}_rgb.tif',
            'source': image_source,
            'export_task_id': task.id
        }
        
    except Exception as e:
        print(f"Error exporting images: {e}")
        return None

def earth_engine_handler(event, context):
    """Cloud Function entry point for Earth Engine processing."""
    try:
        # Parse the Pub/Sub message
        pubsub_message = base64.b64decode(event['data']).decode('utf-8')
        enriched_event = json.loads(pubsub_message)
        
        geocoded_data = enriched_event.get('geocoded_data')
        if not geocoded_data:
            print("No geocoded data found in event")
            return
        
        # Get viewport for bounding box
        viewport = geocoded_data.get('viewport')
        if not viewport:
            print("No viewport found in geocoded data")
            return
        
        # Create region geometry
        region = get_bounding_box(viewport)
        
        # Get satellite imagery
        image, image_source = get_satellite_images(region, enriched_event['disaster_type'])
        
        if image:
            # Export imagery to Cloud Storage
            export_meta = export_images_to_storage(
                image, 
                geocoded_data['name'], 
                enriched_event['disaster_type'],
                image_source
            )
            
            if export_meta:
                # Add export metadata to the event
                enriched_event['image_data'] = export_meta
                
                # Publish to next topic for damage assessment
                output_data = json.dumps(enriched_event).encode('utf-8')
                publisher.publish(output_topic_path, data=output_data)
                
                print(f"Successfully processed Earth Engine imagery for {geocoded_data['name']}")
            else:
                print(f"Failed to export Earth Engine imagery for {geocoded_data['name']}")
        else:
            print(f"No suitable imagery found for {geocoded_data['name']}")
        
    except Exception as e:
        print(f"Error in Earth Engine function: {e}")
EOF

cat > $WORK_DIR/earth-engine/requirements.txt << 'EOF'
google-cloud-pubsub
google-cloud-storage
earthengine-api
EOF

# Authenticate Earth Engine
echo -e "${YELLOW}Authenticating Earth Engine...${NC}"
earthengine authenticate --quiet
gcloud functions describe earth-engine-collector --region=$REGION > /dev/null 2>&1 || \
gcloud functions deploy earth-engine-collector \
    --region=$REGION \
    --runtime=python310 \
    --source=$WORK_DIR/earth-engine \
    --entry-point=earth_engine_handler \
    --trigger-topic=geocoded-locations \
    --timeout=540s \
    --memory=1024MB \
    --set-env-vars=GCP_PROJECT=$PROJECT_ID,STORAGE_BUCKET=$STORAGE_BUCKET

# Set up the Damage Assessment function
echo -e "${YELLOW}Setting up Damage Assessment function...${NC}"
mkdir -p $WORK_DIR/damage-assessment
cat > $WORK_DIR/damage-assessment/main.py << 'EOF'
# Insert main.py content for damage-assessment here
# main.py
import base64
import json
import os
import tempfile
import requests
from google.cloud import storage
from google.cloud import pubsub_v1
from google.cloud import bigquery
import rasterio
from rasterio.enums import Resampling
from PIL import Image
import numpy as np
import io

# Set up GCP clients
publisher = pubsub_v1.PublisherClient()
output_topic_path = publisher.topic_path(os.environ.get('GCP_PROJECT'), 'damage-assessment-results')
storage_client = storage.Client()
bq_client = bigquery.Client()

# Roboflow API settings
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY')
ROBOFLOW_PROJECT = os.environ.get('ROBOFLOW_PROJECT', 'disaster-damage-assessment')
ROBOFLOW_VERSION = os.environ.get('ROBOFLOW_VERSION', '1')
ROBOFLOW_SIZE = 640  # Default model input size

def download_image(bucket_name, image_path):
    """Download image from Cloud Storage."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(image_path)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            return temp_file.name
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def process_geotiff(file_path):
    """Process GeoTIFF image and convert to patches for model inference."""
    try:
        patches = []
        coords = []
        
        with rasterio.open(file_path) as src:
            # Get image dimensions
            height = src.height
            width = src.width
            
            # Define patch size (in pixels)
            patch_size = 640
            
            # Calculate overlap to ensure we cover the entire image
            overlap = patch_size // 4
            
            # Calculate steps
            x_steps = max(1, (width - overlap) // (patch_size - overlap))
            y_steps = max(1, (height - overlap) // (patch_size - overlap))
            
            # Extract patches
            for y in range(y_steps):
                for x in range(x_steps):
                    # Calculate patch coordinates
                    x_start = min(width - patch_size, x * (patch_size - overlap))
                    y_start = min(height - patch_size, y * (patch_size - overlap))
                    
                    # Read patch with all bands
                    window = rasterio.windows.Window(x_start, y_start, patch_size, patch_size)
                    patch_data = src.read(window=window)
                    
                    # Normalize and convert to RGB
                    rgb_data = patch_data[:3]  # Assuming RGB bands are first 3
                    
                    # Normalize each band separately
                    for i in range(rgb_data.shape[0]):
                        band = rgb_data[i]
                        min_val = np.percentile(band, 2)
                        max_val = np.percentile(band, 98)
                        
                        # Avoid division by zero
                        if max_val > min_val:
                            rgb_data[i] = np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
                        else:
                            rgb_data[i] = np.zeros_like(band, dtype=np.uint8)
                    
                    # Convert to RGB image
                    rgb_image = np.transpose(rgb_data, (1, 2, 0))
                    pil_image = Image.fromarray(rgb_image)
                    
                    # Save coordinates of this patch
                    coords.append({
                        'x_start': x_start,
                        'y_start': y_start,
                        'x_end': x_start + patch_size,
                        'y_end': y_start + patch_size
                    })
                    
                    # Convert to bytes for API
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    
                    patches.append(img_byte_arr.getvalue())
        
        return patches, coords
    
    except Exception as e:
        print(f"Error processing GeoTIFF: {e}")
        return [], []

def run_damage_assessment(image_bytes):
    """Send image to Roboflow for damage assessment."""
    try:
        # API URL for Roboflow
        url = f"https://detect.roboflow.com/{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}"
        
        params = {
            "api_key": ROBOFLOW_API_KEY,
            "confidence": 40,  # Minimum confidence threshold
            "overlap": 30      # NMS overlap threshold
        }
        
        # Send request to Roboflow API
        response = requests.post(
            url,
            params=params,
            data=image_bytes,
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error from Roboflow API: {response.status_code}, {response.text}")
            return None
    
    except Exception as e:
        print(f"Error in damage assessment: {e}")
        return None

def save_results_to_bigquery(results, event):
    """Save damage assessment results to BigQuery."""
    table_id = f"{os.environ.get('GCP_PROJECT')}.disaster_data.damage_assessments"
    
    # Ensure the dataset and table exist
    try:
        dataset_ref = bigquery.DatasetReference(os.environ.get('GCP_PROJECT'), 'disaster_data')
        bq_client.get_dataset(dataset_ref)
    except Exception:
        bq_client.create_dataset('disaster_data')
        
    schema = [
        bigquery.SchemaField("disaster_type", "STRING"),
        bigquery.SchemaField("location_name", "STRING"),
        bigquery.SchemaField("damage_class", "STRING"),
        bigquery.SchemaField("confidence", "FLOAT"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("news_title", "STRING"),
        bigquery.SchemaField("image_path", "STRING"),
        bigquery.SchemaField("latitude", "FLOAT"),
        bigquery.SchemaField("longitude", "FLOAT")
    ]
    
    table = bigquery.Table(table_id, schema=schema)
    try:
        bq_client.get_table(table)
    except Exception:
        bq_client.create_table(table)
    
    # Get event data
    geocoded_data = event['geocoded_data']
    location_name = geocoded_data['name']
    disaster_type = event['disaster_type']
    news_title = event['title']
    image_path = event['image_data']['path']
    lat = geocoded_data['latitude']
    lng = geocoded_data['longitude']
    
    # Process predictions
    rows_to_insert = []
    
    for damage_class, predictions in results.items():
        for pred in predictions:
            row = {
                "disaster_type": disaster_type,
                "location_name": location_name,
                "damage_class": damage_class,
                "confidence": pred['confidence'],
                "timestamp": event['timestamp'],
                "news_title": news_title,
                "image_path": image_path,
                "latitude": lat,
                "longitude": lng
            }
            rows_to_insert.append(row)
    
    if rows_to_insert:
        errors = bq_client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            print(f"Errors inserting rows: {errors}")
        else:
            print(f"Inserted {len(rows_to_insert)} damage assessment results to BigQuery")
def damage_assessment_handler(event, context):
    """Cloud Function entry point for damage assessment."""
    try:
        # Parse the Pub/Sub message
        pubsub_message = base64.b64decode(event['data']).decode('utf-8')
        enriched_event = json.loads(pubsub_message)
        
        image_data = enriched_event.get('image_data')
        if not image_data:
            print("No image data found in event")
            return
        
        # Download image
        bucket_name = image_data['bucket']
        image_path = image_data['path']
        local_file = download_image(bucket_name, image_path)
        
        if not local_file:
            print("Failed to download image")
            return
        
        # Process GeoTIFF into patches
        image_patches, patch_coords = process_geotiff(local_file)
        
        if not image_patches:
            print("No valid image patches created")
            os.unlink(local_file)
            return
        
        # Run damage assessment on each patch
        all_results = {'no-damage': [], 'minor-damaged': [], 'major-damaged': [], 'destroyed': []}
        
        for i, patch in enumerate(image_patches):
            # Send patch to Roboflow
            result = run_damage_assessment(patch)
            
            if result and 'predictions' in result:
                # Process predictions
                for pred in result['predictions']:
                    # Get class name and confidence
                    class_name = pred['class']
                    confidence = pred['confidence']
                    
                    # Get bounding box in patch coordinates
                    x = pred['x']
                    y = pred['y']
                    width = pred['width']
                    height = pred['height']
                    
                    # Convert to image coordinates
                    x_offset = patch_coords[i]['x_start']
                    y_offset = patch_coords[i]['y_start']
                    
                    global_x = x_offset + x
                    global_y = y_offset + y
                    
                    # Add to results
                    detection = {
                        'confidence': confidence,
                        'x': global_x,
                        'y': global_y,
                        'width': width,
                        'height': height,
                        'patch_index': i
                    }
                    
                    if class_name in all_results:
                        all_results[class_name].append(detection)
        
        # Clean up
        os.unlink(local_file)
        
        # Filter for severe damage only
        severe_damage_count = len(all_results['major-damaged']) + len(all_results['destroyed'])
        total_detections = sum(len(detections) for detections in all_results.values())
        
        # Add results to event
        enriched_event['damage_assessment'] = {
            'results': all_results,
            'severe_damage_count': severe_damage_count,
            'total_detections': total_detections,
            'severe_damage_percentage': (severe_damage_count / total_detections * 100) if total_detections > 0 else 0
        }
        
        # Save results to BigQuery
        save_results_to_bigquery(all_results, enriched_event)
        
        # Publish to next topic for visualization
        output_data = json.dumps(enriched_event).encode('utf-8')
        publisher.publish(output_topic_path, data=output_data)
        
        print(f"Completed damage assessment with {severe_damage_count} severe damage detections")
        
    except Exception as e:
        print(f"Error in damage assessment function: {e}")

EOF

cat > $WORK_DIR/damage-assessment/requirements.txt << 'EOF'
google-cloud-pubsub==2.13.5
google-cloud-storage==2.7.0
google-cloud-bigquery==3.3.5
rasterio==1.3.4
Pillow==9.4.0
numpy==1.24.2
requests==2.28.1
EOF
gcloud functions describe damage-assessment--region=$REGION > /dev/null 2>&1 || \
gcloud functions deploy damage-assessment \
    --region=$REGION \
    --runtime=python310 \
    --source=$WORK_DIR/damage-assessment \
    --entry-point=damage_assessment_handler \
    --trigger-topic=satellite-images \
    --timeout=540s \
    --memory=2048MB \
    --set-env-vars=GCP_PROJECT=$PROJECT_ID,ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY,ROBOFLOW_PROJECT=$ROBOFLOW_PROJECT,ROBOFLOW_VERSION=$ROBOFLOW_VERSION

# Set up the Dashboard
echo -e "${YELLOW}Setting up Dashboard...${NC}"
mkdir -p $WORK_DIR/dashboard/templates
cat > $WORK_DIR/dashboard/app.py << 'EOF'
from flask import Flask, jsonify, render_template
from datetime import datetime

app = Flask(__name__)

# Dummy event list
dummy_events = [
    {
        "location": "Ramban, Jammu & Kashmir",
        "disaster_type": "flood",
        "news_title": "Flash Floods in Ramban",
        "news_url": "https://reliefweb.int/report/india/flash-floods-ramban",
        "timestamp": "2025-04-20T14:32:00Z",
        "latitude": 33.25,
        "longitude": 75.25,
        "damage_summary": {
            "no-damage": {"count": 5, "confidence": 0.92},
            "minor-damaged": {"count": 7, "confidence": 0.88},
            "major-damaged": {"count": 12, "confidence": 0.91},
            "destroyed": {"count": 4, "confidence": 0.93}
        }
    },
    {
        "location": "Mangan, Sikkim",
        "disaster_type": "landslide",
        "news_title": "Landslide in North Sikkim",
        "news_url": "https://example.com/sikkim-landslide",
        "timestamp": "2025-04-18T09:00:00Z",
        "latitude": 27.5,
        "longitude": 88.5,
        "damage_summary": {
            "no-damage": {"count": 10, "confidence": 0.95},
            "minor-damaged": {"count": 6, "confidence": 0.89},
            "major-damaged": {"count": 2, "confidence": 0.90},
            "destroyed": {"count": 0, "confidence": 0.00}
        }
    },
    {
        "location": "Sonitpur, Assam",
        "disaster_type": "earthquake",
        "news_title": "Moderate Earthquake in Assam",
        "news_url": "https://example.com/assam-quake",
        "timestamp": "2025-04-10T06:45:00Z",
        "latitude": 26.6,
        "longitude": 92.7,
        "damage_summary": {
            "no-damage": {"count": 20, "confidence": 0.96},
            "minor-damaged": {"count": 3, "confidence": 0.84},
            "major-damaged": {"count": 0, "confidence": 0.00},
            "destroyed": {"count": 0, "confidence": 0.00}
        }
    }
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/recent-events")
def recent_events():
    return jsonify(dummy_events)

@app.route("/api/disaster-stats")
def disaster_stats():
    stats = []
    disaster_types = set([e["disaster_type"] for e in dummy_events])
    for dtype in disaster_types:
        filtered = [e for e in dummy_events if e["disaster_type"] == dtype]
        count = sum((e["damage_summary"].get("major-damaged", {}).get("count", 0) +
                     e["damage_summary"].get("destroyed", {}).get("count", 0)) for e in filtered)
        stats.append({
            "disaster_type": dtype,
            "affected_locations": len(filtered),
            "severe_damage_count": count
        })
    return jsonify(stats)

@app.route("/api/location-details/<location>")
def location_details(location):
    event = next((e for e in dummy_events if e["location"].lower() == location.lower()), None)
    if not event:
        return jsonify({"error": "Location not found"}), 404

    return jsonify({
        "location": event["location"],
        "formatted_address": f"{event['location']}, India",
        "disaster_type": event["disaster_type"],
        "damage_summary": event["damage_summary"],
        "news": [{
            "title": event["news_title"],
            "url": event["news_url"],
            "timestamp": event["timestamp"]
        }]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

EOF

cat > $WORK_DIR/dashboard/templates/index.html << 'EOF'

<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Disaster Assessment Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
    <style>
        #map { height: 500px; }
        .severe-damage { color: #dc2626; }
        .minor-damage { color: #f59e0b; }
        .no-damage { color: #10b981; }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-10">
            <h1 class="text-4xl font-bold text-center text-gray-800">Natural Disaster Damage Assessment</h1>
            <p class="text-xl text-center text-gray-600 mt-2">Real-time monitoring of disaster impacts</p>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-semibold mb-4">Filter Options</h2>
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2">Time Range</label>
                    <select id="time-range" class="w-full border rounded px-3 py-2">
                        <option value="7">Last 7 days</option>
                        <option value="30">Last 30 days</option>
                        <option value="90">Last 90 days</option>
                    </select>
                </div>
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2">Disaster Type</label>
                    <select id="disaster-type" class="w-full border rounded px-3 py-2">
                        <option value="all">All Disasters</option>
                        <option value="earthquake">Earthquake</option>
                        <option value="flood">Flood</option>
                        <option value="hurricane">Hurricane</option>
                        <option value="wildfire">Wildfire</option>
                        <option value="tsunami">Tsunami</option>
                    </select>
                </div>
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2">Damage Level</label>
                    <select id="damage-level" class="w-full border rounded px-3 py-2">
                        <option value="all">All Levels</option>
                        <option value="severe">Severe Damage Only</option>
                    </select>
                </div>
                <button id="apply-filters" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">Apply Filters</button>
            </div>

            <div class="bg-white rounded-lg shadow p-6 lg:col-span-2">
                <h2 class="text-xl font-semibold mb-4">Disaster Impact Overview</h2>
                <canvas id="disaster-chart" height="200"></canvas>
            </div>
        </div>

        <div class="grid grid-cols-1 mb-8">
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-semibold mb-4">Disaster Map</h2>
                <div id="map"></div>
            </div>
        </div>

        <div class="grid grid-cols-1 mb-8">
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-semibold mb-4">Recent Disaster Events</h2>
                <div class="overflow-x-auto">
                    <table class="min-w-full bg-white">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="py-2 px-4 text-left">Location</th>
                                <th class="py-2 px-4 text-left">Disaster Type</th>
                                <th class="py-2 px-4 text-left">News Title</th>
                                <th class="py-2 px-4 text-center">Damage Level</th>
                                <th class="py-2 px-4 text-center">Date</th>
                                <th class="py-2 px-4 text-center">Details</th>
                            </tr>
                        </thead>
                        <tbody id="events-table-body">
                            <!-- Dynamically filled -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Location Detail Modal -->
        <div id="location-modal" class="fixed inset-0 bg-black bg-opacity-50 hidden flex items-center justify-center">
            <div class="bg-white rounded-lg p-8 max-w-2xl w-full max-h-screen overflow-y-auto">
                <div class="flex justify-between items-center mb-6">
                    <h2 id="modal-title" class="text-2xl font-bold"></h2>
                    <button id="close-modal" class="text-gray-500 hover:text-gray-700">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                <div id="modal-content">
                    <!-- Dynamically filled -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let map;
        let markers = [];
        let disasterChart;
        let allEvents = [];

        // Initialize the dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initMap();
            fetchRecentEvents();
            fetchDisasterStats();

            // Event listeners
            document.getElementById('apply-filters').addEventListener('click', applyFilters);
            document.getElementById('close-modal').addEventListener('click', closeLocationModal);
        });

        // Initialize the map
        function initMap() {
            map = L.map('map').setView([20, 0], 2);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);
        }

        // Fetch recent events
        function fetchRecentEvents() {
            const timeRange = document.getElementById('time-range').value;
            
            fetch(`/api/recent-events?days=${timeRange}`)
                .then(response => response.json())
                .then(events => {
                    allEvents = events;
                    displayEvents(events);
                    addMarkersToMap(events);
                })
                .catch(error => console.error('Error fetching events:', error));
        }

        // Fetch disaster statistics
        function fetchDisasterStats() {
            const timeRange = document.getElementById('time-range').value;
            
            fetch(`/api/disaster-stats?days=${timeRange}`)
                .then(response => response.json())
                .then(stats => {
                    displayDisasterChart(stats);
                })
                .catch(error => console.error('Error fetching stats:', error));
        }

        // Apply filters to the data
        function applyFilters() {
            fetchRecentEvents();
            fetchDisasterStats();
        }

        // Display events in the table
        function displayEvents(events) {
            const tableBody = document.getElementById('events-table-body');
            tableBody.innerHTML = '';
            
            events.forEach(event => {
                // Calculate damage level
                let damageLevel = 'Low';
                let damageClass = 'no-damage';
                
                if (event.damage_summary) {
                    const severeCount = (event.damage_summary['major-damaged']?.count || 0) + 
                                        (event.damage_summary['destroyed']?.count || 0);
                    
                    if (severeCount > 10) {
                        damageLevel = 'Severe';
                        damageClass = 'severe-damage';
                    } else if (severeCount > 0) {
                        damageLevel = 'Moderate';
                        damageClass = 'minor-damage';
                    }
                }
                
                // Format date
                const date = new Date(event.timestamp);
                const formattedDate = date.toLocaleDateString();
                
                // Create table row
                const row = document.createElement('tr');
                row.className = 'border-t hover:bg-gray-50';
                row.innerHTML = `
                    <td class="py-3 px-4">${event.location}</td>
                    <td class="py-3 px-4">${capitalizeFirstLetter(event.disaster_type)}</td>
                    <td class="py-3 px-4">
                        <a href="${event.news_url}" target="_blank" class="text-blue-600 hover:underline">${event.news_title}</a>
                    </td>
                    <td class="py-3 px-4 text-center ${damageClass} font-medium">${damageLevel}</td>
                    <td class="py-3 px-4 text-center">${formattedDate}</td>
                    <td class="py-3 px-4 text-center">
                        <button class="view-details bg-blue-100 text-blue-800 px-3 py-1 rounded hover:bg-blue-200" 
                                data-location="${event.location}">View</button>
                    </td>
                `;
                
                tableBody.appendChild(row);
            });
            
            // Add event listeners to detail buttons
            document.querySelectorAll('.view-details').forEach(button => {
                button.addEventListener('click', function() {
                    const location = this.getAttribute('data-location');
                    showLocationDetails(location);
                });
            });
        }

        // Add markers to the map
        function addMarkersToMap(events) {
            // Clear existing markers
            markers.forEach(marker => map.removeLayer(marker));
            markers = [];
            
            events.forEach(event => {
                if (event.latitude && event.longitude) {
                    // Determine marker color based on damage level
                    let markerColor = 'green';
                    
                    if (event.damage_summary) {
                        const severeCount = (event.damage_summary['major-damaged']?.count || 0) + 
                                            (event.damage_summary['destroyed']?.count || 0);
                        
                        if (severeCount > 10) {
                            markerColor = 'red';
                        } else if (severeCount > 0) {
                            markerColor = 'orange';
                        }
                    }
                    
                    // Create marker
                    const marker = L.circleMarker([event.latitude, event.longitude], {
                        radius: 8,
                        fillColor: markerColor,
                        color: '#fff',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    }).addTo(map);
                    
                    // Add popup
                    marker.bindPopup(`
                        <strong>${event.location}</strong><br>
                        ${capitalizeFirstLetter(event.disaster_type)}<br>
                        <a href="#" class="text-blue-600" onclick="showLocationDetails('${event.location}'); return false;">View details</a>
                    `);
                    
                    markers.push(marker);
                }
            });
        }

        // Display disaster chart
        function displayDisasterChart(stats) {
            const ctx = document.getElementById('disaster-chart').getContext('2d');
            
            // Extract data for chart
            const labels = stats.map(stat => capitalizeFirstLetter(stat.disaster_type));
            const affectedLocations = stats.map(stat => stat.affected_locations);
            const severeDamageCount = stats.map(stat => stat.severe_damage_count);
            
            // Destroy existing chart if it exists
            if (disasterChart) {
                disasterChart.destroy();
            }
            
            // Create new chart
            disasterChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Affected Locations',
                            data: affectedLocations,
                            backgroundColor: 'rgba(59, 130, 246, 0.5)',
                            borderColor: 'rgb(59, 130, 246)',
                            borderWidth: 1
                        },
                        {
                            label: 'Severe Damage Instances',
                            data: severeDamageCount,
                            backgroundColor: 'rgba(220, 38, 38, 0.5)',
                            borderColor: 'rgb(220, 38, 38)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        // Show location details in modal
        function showLocationDetails(location) {
            fetch(`/api/location-details/${encodeURIComponent(location)}`)
                .then(response => response.json())
                .then(details => {
                    // Update modal title
                    document.getElementById('modal-title').textContent = details.location;
                    
                    // Create content
                    let content = `
                        <div class="mb-4">
                            <p class="text-gray-600">${details.formatted_address}</p>
                            <p class="mt-2"><span class="font-semibold">Disaster Type:</span> ${capitalizeFirstLetter(details.disaster_type)}</p>
                        </div>
                        
                        <div class="mb-6">
                            <h3 class="text-lg font-semibold mb-2">Damage Assessment</h3>
                            <div class="grid grid-cols-2 gap-4">
                    `;
                    
                    // Damage summary
                    const damageClasses = ['no-damage', 'minor-damaged', 'major-damaged', 'destroyed'];
                    damageClasses.forEach(damageClass => {
                        const count = details.damage_summary[damageClass]?.count || 0;
                        const confidence = details.damage_summary[damageClass]?.confidence || 0;
                        
                        let colorClass = '';
                        if (damageClass === 'major-damaged' || damageClass === 'destroyed') {
                            colorClass = 'severe-damage';
                        } else if (damageClass === 'minor-damaged') {
                            colorClass = 'minor-damage';
                        } else {
                            colorClass = 'no-damage';
                        }
                        
                        content += `
                            <div class="bg-gray-50 p-3 rounded">
                                <p class="font-semibold ${colorClass}">${formatDamageClass(damageClass)}</p>
                                <p>Count: ${count}</p>
                                <p>Avg. Confidence: ${(confidence * 100).toFixed(1)}%</p>
                            </div>
                        `;
                    });
                    
                    content += `
                            </div>
                        </div>
                        
                        <div class="mb-4">
                            <h3 class="text-lg font-semibold mb-2">News Sources</h3>
                            <ul class="list-disc list-inside space-y-2">
                    `;
                    
                    // News sources
                    details.news.forEach(news => {
                        const date = new Date(news.timestamp);
                        content += `
                            <li>
                                <a href="${news.url}" target="_blank" class="text-blue-600 hover:underline">${news.title}</a>
                                <span class="text-gray-500 text-sm ml-2">${date.toLocaleDateString()}</span>
                            </li>
                        `;
                    });
                    
                    content += `
                            </ul>
                        </div>
                    `;
                    
                    // Update modal content
                    document.getElementById('modal-content').innerHTML = content;
                    
                    // Show modal
                    document.getElementById('location-modal').classList.remove('hidden');
                })
                .catch(error => console.error('Error fetching location details:', error));
        }

        // Close the location detail modal
        function closeLocationModal() {
            document.getElementById('location-modal').classList.add('hidden');
        }

        // Helper function to capitalize first letter
        function capitalizeFirstLetter(string) {
            return string.charAt(0).toUpperCase() + string.slice(1);
        }

        // Helper function to format damage class name
        function formatDamageClass(damageClass) {
            return damageClass.split('-').map(capitalizeFirstLetter).join(' ');
        }
    </script>
</body>
</html>
EOF

cat > $WORK_DIR/dashboard/requirements.txt << 'EOF'
flask>=2.0.0,<2.3.0
werkzeug>=2.0.0,<2.3.0
google-cloud-bigquery>=3.0.0
numpy>=1.20.0
gunicorn>=20.0.0
EOF

# Create a Dockerfile for the dashboard
cat > $WORK_DIR/dashboard/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
EOF

# Deploy dashboard to Cloud Run
gcloud builds submit --tag gcr.io/$PROJECT_ID/disaster-dashboard $WORK_DIR/dashboard
gcloud run deploy disaster-dashboard \
    --image gcr.io/$PROJECT_ID/disaster-dashboard \
    --platform managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=512Mi \
    --set-env-vars=GCP_PROJECT=$PROJECT_ID

# Clean up temporary files
echo -e "${YELLOW}Cleaning up temporary files...${NC}"
rm -rf $WORK_DIR

# Get the dashboard URL
DASHBOARD_URL=$(gcloud run services describe disaster-dashboard --platform managed --region=$REGION --format="value(status.url)")

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "Dashboard URL: ${DASHBOARD_URL}"
echo -e "${YELLOW}Note: It may take some time for data to start flowing through the system.${NC}"
echo -e "${YELLOW}Replace the API keys in the deployment script with your actual keys for production use.${NC}"