import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET

# Set up the page
st.set_page_config(
    page_title="AI Disaster Detection Dashboard",
    page_icon="⚠️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stAlert {
        border-left: 5px solid #ff4b4b;
    }
    .disaster-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .earthquake { background-color: #ffdddd; }
    .flood { background-color: #d4e6ff; }
    .wildfire { background-color: #ffedd4; }
    .storm { background-color: #e6e6ff; }
    .landslide { background-color: #ddffdd; }
    .gdacs { border-left: 5px solid #4285F4; }
    .twitter { border-left: 5px solid #34A853; }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🌍 AI Disaster Detection Dashboard")
st.markdown("""
    Real-time monitoring of global disasters combining official reports and social media alerts.
    """)

# Data loading functions
@st.cache_data(ttl=300)
def load_twitter_data():
    try:
        twitter_df = pd.read_csv("content/processed_disaster_tweets.csv", parse_dates=['created_at'])
        twitter_df = twitter_df.rename(columns={'created_at': 'timestamp'})
        twitter_df['reported_by'] = 'Twitter'
        twitter_df['confidence'] = 0.85  # Default confidence for social media
        return twitter_df
    except Exception as e:
        st.error(f"Error loading Twitter data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_gdacs_data():
    try:
        url = "https://www.gdacs.org/xml/rss.xml"
        response = requests.get(url)
        root = ET.fromstring(response.content)
        disasters = []

        event_type_map = {
            "EQ": "Earthquake", "FL": "Flood", "WF": "Wildfire",
            "TC": "Storm", "VO": "Volcano", "LS": "Landslide", 
            "DR": "Drought", "OT": "Other"
        }

        severity_map = {"Green": 3, "Orange": 6, "Red": 9}

        for item in root.findall(".//item"):
            title = item.find("title").text
            description = item.find("description").text
            lat = float(item.find("{http://www.georss.org/georss}point").text.split()[0])
            lon = float(item.find("{http://www.georss.org/georss}point").text.split()[1])
            pub_date = pd.to_datetime(item.find("pubDate").text)

            event_type_code = item.find("{http://www.gdacs.org}eventtype").text
            alert_level = item.find("{http://www.gdacs.org}alertlevel").text

            disasters.append({
                "id": f"GDACS-{item.find('{http://www.gdacs.org}eventid').text}",
                "type": event_type_map.get(event_type_code, "Other"),
                "latitude": lat,
                "longitude": lon,
                "severity": severity_map.get(alert_level, 5),
                "description": f"{title}: {description}",
                "timestamp": pub_date,
                "reported_by": "GDACS",
                "confidence": 0.95,
                "country": item.find("{http://www.gdacs.org}country").text or "Unknown",
                "population": item.find("{http://www.gdacs.org}population").get("value", "N/A")
            })

        return pd.DataFrame(disasters)
    except Exception as e:
        st.error(f"Error fetching GDACS data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_combined_data():
    twitter_df = load_twitter_data()
    gdacs_df = fetch_gdacs_data()
    
    if not twitter_df.empty and not gdacs_df.empty:
        return pd.concat([twitter_df, gdacs_df], ignore_index=True)
    elif not twitter_df.empty:
        return twitter_df
    elif not gdacs_df.empty:
        return gdacs_df
    else:
        return pd.DataFrame()

# Load data
combined_df = load_combined_data()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    available_types = combined_df['type'].unique().tolist() if not combined_df.empty else []
    disaster_types = st.multiselect(
        "Disaster Types",
        available_types,
        default=["Earthquake", "Flood", "Wildfire"] if available_types else []
    )
    
    severity_levels = st.slider(
        "Severity Level",
        1, 10, (3, 10)
    )
    
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 hours", "Last 7 days", "Last 30 days"],
        index=0
    )
    
    source_types = st.multiselect(
        "Data Sources",
        ["GDACS", "Twitter"],
        default=["GDACS", "Twitter"]
    )

# Apply filters
if not combined_df.empty:
    filtered_df = combined_df[
        (combined_df["type"].isin(disaster_types)) & 
        (combined_df["severity"] >= severity_levels[0]) & 
        (combined_df["severity"] <= severity_levels[1]) &
        (combined_df["reported_by"].isin(source_types))
    ].copy()
    
    if time_range == "Last 24 hours":
        filtered_df = filtered_df[filtered_df['timestamp'] > datetime.now() - timedelta(hours=24)]
    elif time_range == "Last 7 days":
        filtered_df = filtered_df[filtered_df['timestamp'] > datetime.now() - timedelta(days=7)]
    elif time_range == "Last 30 days":
        filtered_df = filtered_df[filtered_df['timestamp'] > datetime.now() - timedelta(days=30)]
else:
    filtered_df = pd.DataFrame()

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Disasters", len(filtered_df))
col2.metric("Highest Severity", filtered_df["severity"].max() if not filtered_df.empty else 0)
col3.metric("Most Common", filtered_df["type"].mode()[0] if not filtered_df.empty else "N/A")
col4.metric("New in Last Hour", 
           len(filtered_df[filtered_df["timestamp"] > datetime.now() - timedelta(hours=1)]) if not filtered_df.empty else 0)

# Main content
tab1, tab2, tab3 = st.tabs(["Map View", "Data Analysis", "Alert Feed"])

with tab1:
    st.subheader("Global Disaster Map")
    
    if not filtered_df.empty:
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        color_map = {
            "Earthquake": "red",
            "Flood": "blue",
            "Wildfire": "orange",
            "Storm": "purple",
            "Landslide": "green",
            "Volcano": "black",
            "Drought": "beige"
        }
        
        for idx, row in filtered_df.iterrows():
            icon_color = "darkblue" if row['reported_by'] == 'GDACS' else "darkgreen"
            
            popup_text = f"""
                <b>Type:</b> {row['type']}<br>
                <b>Severity:</b> {row['severity']}/10<br>
                <b>Source:</b> {row['reported_by']}<br>
                <small>{row['description'][:100]}...</small>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                icon=folium.Icon(
                    color=color_map.get(row['type'], 'gray'),
                    icon='info-sign' if row['severity'] > 7 else 'warning-sign'
                )
            ).add_to(m)
        
        folium_static(m, width=1200, height=600)
    else:
        st.warning("No disasters match your current filters.")

with tab2:
    st.subheader("Data Analysis")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.pie(
                filtered_df, 
                names='type', 
                title='Disaster Type Distribution',
                hole=0.4,
                color='type',
                color_discrete_map={
                    "Earthquake": "#ff1100",
                    "Flood": "#d4e6ff",
                    "Wildfire": "orange",
                    "Storm": "#e6e6ff",
                    "Landslide": "#ddffdd"
                }
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.histogram(
                filtered_df, 
                x='severity', 
                nbins=10,
                title='Severity Distribution',
                color='type',
                color_discrete_map={
                    "Earthquake": "#ff1100",
                    "Flood": "#d4e6ff",
                    "Wildfire": "orange",
                    "Storm": "#e6e6ff",
                    "Landslide": "#ddffdd"
                }
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        time_series = filtered_df.set_index('timestamp').resample('6H').agg({
            'severity': 'mean',
            'type': 'count'
        }).rename(columns={'type': 'count'}).reset_index()
        
        fig3 = px.line(
            time_series,
            x='timestamp',
            y=['count', 'severity'],
            title='Disasters Over Time',
            labels={'value': 'Count/Mean Severity'},
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No data to display with current filters.")

with tab3:
    st.subheader("Recent Alerts")
    
    if not filtered_df.empty:
        recent_alerts = filtered_df.sort_values('timestamp', ascending=False).head(10)
        
        for _, alert in recent_alerts.iterrows():
            alert_time = alert['timestamp'].strftime('%Y-%m-%d %H:%M UTC')
            source_class = "gdacs" if alert['reported_by'] == 'GDACS' else "twitter"
            
            st.markdown(f"""
                <div class="disaster-card {alert['type'].lower()} {source_class}">
                    <div style="display: flex; justify-content: space-between;">
                        <h4>{alert['type']} Alert ⚠️</h4>
                        <small>Source: {alert['reported_by']}</small>
                    </div>
                    <p><b>Location:</b> {alert.get('country', alert.get('location', 'Unknown'))}</p>
                    <p><b>Severity:</b> {alert['severity']}/10 | <b>Confidence:</b> {alert['confidence']*100:.0f}%</p>
                    <p>{alert['description'][:200]}...</p>
                    <p><small><b>Detected:</b> {alert_time}</small></p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No recent alerts match your filters.")

# Refresh button
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Last updated time
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")