from dotenv import load_dotenv
import sqlite3
import tweepy
import os
import time

load_dotenv()

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if BEARER_TOKEN is None:
    raise ValueError("TWITTER_BEARER_TOKEN is not set. Check your .env file.")

client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Optimized query: Only high-impact disaster terms
query = "(earthquake OR floods OR bombing OR explosion OR hurricane OR tornado) -is:retweet lang:en"    #OR tsunami OR wildfire OR flood 

def save_to_db(tweet_id, tweet_text):
    """Save tweets to SQLite database."""
    conn = sqlite3.connect("tweets.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO disaster_tweets (tweet_id, tweet_text) VALUES (?, ?)", (tweet_id, tweet_text))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Tweet {tweet_id} already exists.")

    conn.close()

def fetch_tweets():
    """Fetch tweets and store them in the database."""
    tweets = client.search_recent_tweets(query=query, max_results=10)
    
    if tweets.data:
        for tweet in tweets.data:
            save_to_db(tweet.id, tweet.text)
        print("✅ Tweets saved!")
    else:
        print("No new tweets found.")

    if tweets.data is None: 
        return []

    return tweets

print("Fetching tweets :" ,fetch_tweets())