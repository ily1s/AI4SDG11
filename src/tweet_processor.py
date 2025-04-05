import sqlite3
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def geocode_location(location, geolocator, geocode):
    """Convert location text to lat/long coordinates"""
    if not location or location == 'Unknown':
        return None, None
    
    try:
        location_data = geocode(location, exactly_one=True)
        if location_data:
            return location_data.latitude, location_data.longitude
    except Exception as e:
        logger.warning(f"Geocoding failed for '{location}': {str(e)}")
        time.sleep(1)  # Rate limiting
    return None, None

def process_tweets_from_db():
    """Fetch, process, and geocode tweets from database"""
    try:
        # Initialize geocoder
        geolocator = Nominatim(user_agent="disaster_dashboard")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        # Connect to database
        conn = sqlite3.connect("tweets.db")
        query = """
        SELECT tweet_id, tweet_text, disaster_type, timestamp, location 
        FROM disaster_tweets 
        WHERE disaster_type IS NOT NULL
        """
        raw_df = pd.read_sql_query(query, conn)
        conn.close()

        if raw_df.empty:
            logger.warning("No disaster tweets found in database")
            return False

        # Process into dashboard format
        processed_df = pd.DataFrame({
            'id': raw_df['tweet_id'],
            'text': raw_df['tweet_text'],
            'type': raw_df['disaster_type'],
            'location': raw_df['location'].fillna('Unknown'),
            'timestamp': pd.to_datetime(raw_df['timestamp']),
            'hashtags': [[]] * len(raw_df),
            'retweets': 0,
            'favorites': 0
        })

        # Geocode locations
        processed_df['latitude'] = None
        processed_df['longitude'] = None
        
        unique_locations = processed_df['location'].unique()
        location_coords = {}
        
        for loc in unique_locations:
            lat, lon = geocode_location(loc, geolocator, geocode)
            if lat and lon:
                location_coords[loc] = (lat, lon)
        
        # Apply coordinates
        for loc, (lat, lon) in location_coords.items():
            mask = processed_df['location'] == loc
            processed_df.loc[mask, 'latitude'] = lat
            processed_df.loc[mask, 'longitude'] = lon

        # Calculate severity (customize as needed)
        processed_df['severity'] = processed_df.apply(
            lambda x: min(10, 3 + len(x['text'])//100),  # Example calculation
            axis=1
        )

        # Save to CSV
        processed_df.to_csv("content/processed_disaster_tweets.csv", index=False)
        logger.info(f"Processed {len(processed_df)} tweets with {len(location_coords)} geocoded locations")
        return True

    except Exception as e:
        logger.error(f"Error processing tweets: {e}")
        return False

if __name__ == "__main__":
    process_tweets_from_db()