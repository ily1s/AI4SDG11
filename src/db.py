import sqlite3
import pandas as pd
# Connect to SQLite
conn = sqlite3.connect("tweets.db")
cursor = conn.cursor()

# # Create table for tweets (if not exists)
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS disaster_tweets (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     tweet_id TEXT UNIQUE,
#     tweet_text TEXT,
#     disaster_type TEXT DEFAULT NULL,
#     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
# )
# """)

# Check if 'location' column exists
# cursor.execute("PRAGMA table_info(disaster_tweets)")
# columns = [col[1] for col in cursor.fetchall()]  
# if "location" not in columns:
#     cursor.execute("ALTER TABLE disaster_tweets ADD COLUMN location TEXT")
#     conn.commit()
#     print("✅ Added 'location' column to disaster_tweets table.")
# else:
#     print("✅ 'location' column already exists.")
    
# cursor.execute("DELETE FROM disaster_tweets")   

query = "select tweet_text, disaster_type, location from disaster_tweets"
df=pd.read_sql_query(query, conn)
print(df)

conn.commit()
conn.close()
