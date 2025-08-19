import streamlit as st
import requests
import pandas as pd
import numpy as np
import itertools
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import folium
from streamlit_folium import folium_static
import json
import os

st.set_page_config(layout="wide")

geolocator = Nominatim(user_agent="route_finder", timeout=10)


@st.cache_data
def load_data():
    return pd.read_excel('Uk_data.xlsx')


df = load_data()

if 'Location' not in df.columns:
    df['Location'] = df['district_name'].astype(str) + ', ' + df['state_name'].astype(str)



def load_geo_cache():
    if os.path.exists('geo_cache.json'):
        with open('geo_cache.json', 'r') as f:
            return json.load(f)
    return {}


def save_geo_cache(cache):
    with open('geo_cache.json', 'w') as f:
        json.dump(cache, f)


geo_cache = load_geo_cache()


def safe_geocode(location_name):
    if location_name in geo_cache:
        return geo_cache[location_name]

    try:
        location = geolocator.geocode(location_name)
        if location:
            lat_lon = (location.latitude, location.longitude)
            geo_cache[location_name] = lat_lon
            save_geo_cache(geo_cache)
            return lat_lon
    except Exception:
        pass
    geo_cache[location_name] = (None, None)
    save_geo_cache(geo_cache)
    return (None, None)


@st.cache_data
def add_coordinates(df):
    lats, lons = [], []
    for loc in df['Location']:
        lat, lon = safe_geocode(loc)
        lats.append(lat)
        lons.append(lon)
    df['Latitude'] = lats
    df['Longitude'] = lons
    return df


if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
    df = add_coordinates(df)



@st.cache_data
def compute_crime_scores(df):
    crime_cols = [
        'murder', 'rape', 'robbery', 'atmpt_cmmt_murder',
        'sex_hrrsmt_work_office_prms', 'voyeurism', 'stalking',
        'auto_motor_vehicle_theft', 'other_thefts', 'dacoity',
        'kidnapping_for_ransom', 'kidnp_abdctn_marrg', 'missing_child_kidnpd',
        'rioting_communal_religious', 'riot_polc_prsnl_gvt_srvnt'
    ]
    weights = {
        'murder': 15.0, 'rape': 12.5, 'robbery': 2.0, 'atmpt_cmmt_murder': 2.2,
        'sex_hrrsmt_work_office_prms': 1.5, 'voyeurism': 1.2, 'stalking': 1.2,
        'auto_motor_vehicle_theft': 1.8, 'other_thefts': 1.0, 'dacoity': 2.2,
        'kidnapping_for_ransom': 2.5, 'kidnp_abdctn_marrg': 2.0, 'missing_child_kidnpd': 2.0,
        'rioting_communal_religious': 1.8, 'riot_polc_prsnl_gvt_srvnt': 2.0
    }
    df.fillna(0, inplace=True)
    df['Crime_Score_Raw'] = sum(df[col] * weights[col] for col in crime_cols)
    scaler = MinMaxScaler((0, 10))
    df['Crime_Score'] = scaler.fit_transform(df[['Crime_Score_Raw']])
    return df


df = compute_crime_scores(df)

location_scores = dict(zip(df['Location'], df['Crime_Score']))
location_coords = dict(zip(df['Location'], zip(df['Latitude'], df['Longitude'])))


@st.cache_data
def train_model(df):
    location_pairs = list(itertools.combinations(df['Location'], 2))
    df_pairs = pd.DataFrame([{
        'From': a, 'To': b, 'Crime_Score': (location_scores[a] + location_scores[b]) / 2
    } for a, b in location_pairs])
    le = LabelEncoder().fit(df['Location'])
    df_pairs['From_encoded'] = le.transform(df_pairs['From'])
    df_pairs['To_encoded'] = le.transform(df_pairs['To'])
    X = df_pairs[['From_encoded', 'To_encoded']]
    y = df_pairs['Crime_Score']
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    return model, le


model, le = train_model(df)


def get_danger_level(score):
    if score < 3:
        return " Safe"
    elif score < 5:
        return " Mild"
    elif score < 7:
        return " Be Aware"
    elif score < 9:
        return " High Risk"
    return " Alert"


@st.cache_data
def fetch_routes(start_coords, end_coords, max_routes=2):
    url = f"http://router.project-osrm.org/route/v1/driving/{start_coords};{end_coords}?overview=full&geometries=geojson&alternatives=true"
    response = requests.get(url)
    data = response.json()
    if "routes" not in data or not data["routes"]:
        return []
    return data["routes"][:max_routes]


def get_safety_optimized_routes(start, end, max_routes=2, sample_rate=20):
    start_location = geolocator.geocode(start)
    end_location = geolocator.geocode(end)
    if not start_location or not end_location:
        st.error("Unable to fetch coordinates for start or end location.")
        return None
    start_coords = f"{start_location.longitude},{start_location.latitude}"
    end_coords = f"{end_location.longitude},{end_location.latitude}"

    routes = fetch_routes(start_coords, end_coords, max_routes)
    all_routes = []

    for route_idx, route in enumerate(routes):
        coords = route["geometry"]["coordinates"]
        sampled_coords = coords[::sample_rate]
        danger_scores = []
        path_points = []
        cities_covered = []

        for lng, lat in sampled_coords:
            route_point = (lat, lng)
            try:
                nearest_city = min(
                    location_coords.items(),
                    key=lambda x: geodesic(route_point, x[1]).km
                )[0]
            except:
                continue
            if nearest_city not in cities_covered:
                cities_covered.append(nearest_city)
                try:
                    enc = le.transform([nearest_city])[0]
                    score = model.predict([[enc, enc]])[0]
                except:
                    score = 5.0
                danger_scores.append(score)
            path_points.append((route_point, score))

        avg_danger = np.mean(danger_scores) if danger_scores else 5.0
        total_distance = route["distance"] / 1000
        all_routes.append({
            'route_number': route_idx + 1,
            'average_danger': avg_danger,
            'total_distance': total_distance,
            'danger_points': path_points,
            'cities_covered': list(cities_covered)
        })

    return sorted(all_routes, key=lambda x: x['average_danger'])


st.title("Route Safety Advisor")
st.write("Evaluate safety scores of routes between two locations.")

start = st.text_input("Start Location", value="")
end = st.text_input("End Location", value="")

max_routes = 2
sample_rate =10

if 'routes' not in st.session_state:
    st.session_state.routes = []

if st.button("Find Safe Routes"):
    with st.spinner("Calculating routes and safety scores..."):
        routes = get_safety_optimized_routes(start, end, max_routes, sample_rate)
        if routes:
            st.session_state.routes = routes
        else:
            st.error("No routes found or error occurred.")

routes = st.session_state.routes
if routes:
    st.success(f"Found {len(routes)} route option(s).")
    
    for i, route in enumerate(routes):
        st.write(f"### Route {i + 1}")
        st.write(f"Safety Score: {route['average_danger']:.2f}/10")
        st.write(f"Distance: {route['total_distance']:.2f} km")
        st.write(f"Cities Covered: {', '.join(route['cities_covered'])}")

    def visualize_routes(routes):
        m = folium.Map(location=routes[0]['danger_points'][0][0], zoom_start=7)
        colors = ['blue', 'green', 'purple', 'orange', 'darkred']
        for idx, route in enumerate(routes):
            folium.PolyLine(
                locations=[p[0] for p in route['danger_points']],
                color=colors[idx % len(colors)],
                weight=2.5,
                opacity=0.7,
                tooltip=f"Route {idx + 1} | Safety Score: {route['average_danger']:.2f}"
            ).add_to(m)
            for point in [0, -1]:
                folium.Marker(
                    location=route['danger_points'][point][0],
                    icon=folium.Icon(color='red' if point == 0 else 'darkgreen')
                ).add_to(m)
        folium_static(m)

    def visualize_route_on_map(danger_points):
        m = folium.Map(location=danger_points[0][0], zoom_start=10, tiles="OpenStreetMap")
        for loc, score in danger_points:
            folium.CircleMarker(
                location=loc,
                radius=5,
                color="red" if score > 7 else "orange" if score > 5 else "green",
                fill=True,
                popup=f"Score: {score:.2f} → {get_danger_level(score)}"
            ).add_to(m)
        folium.PolyLine([p[0] for p in danger_points], color="blue").add_to(m)
        folium_static(m)

    visualize_routes(routes)

    selected = st.number_input("Select route number to view detailed safety analysis:", min_value=1,
                               max_value=len(routes), value=1)

    st.write(f"### Detailed Safety Analysis for Route {selected}")
    visualize_route_on_map(routes[selected - 1]['danger_points'])


    selected_route = routes[selected - 1]
    coordinates = [pt[0] for pt in selected_route['danger_points']]
    start_lat, start_lng = coordinates[0]
    end_lat, end_lng = coordinates[-1]
    via_coords = coordinates[1:-1][::max(len(coordinates) // 10, 1)]

    waypoints = "/".join([f"{lat},{lng}" for lat, lng in via_coords])
    maps_url = f"https://www.google.com/maps/dir/{start_lat},{start_lng}/{waypoints}/{end_lat},{end_lng}"

    st.markdown("### 🚦 Start Live Navigation")
    if st.button("Open in Google Maps"):
        st.markdown(f"[Click here to open Google Maps 🚗]({maps_url})", unsafe_allow_html=True)
