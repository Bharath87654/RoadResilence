import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from geopy.geocoders import Nominatim
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(layout="wide", page_title="AI Traffic Manager - Road Graph Extraction")

st.title("🚦 Phase I & II: Extraction & Topological Healing")
st.markdown("Convert raw satellite imagery into a queryable routing graph.")

# --- SIDEBAR TUNING CONTROLS ---
st.sidebar.header("🌍 Map Navigation")
search_query = st.sidebar.text_input("Search for a City:", "Hyderabad, India")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Graph Completion Controls")
st.sidebar.caption("Adjust distance and orientation tolerances for dead-end healing.")
healing_threshold = st.sidebar.slider("MAX_DISTANCE (px)", 5.0, 100.0, 20.0, step=5.0)
max_angle_error = st.sidebar.slider("ANGLE_THRESHOLD (°)", 5.0, 90.0, 30.0, step=5.0)

# --- SESSION STATE ---
if "bbox" not in st.session_state:
    st.session_state.bbox = None


@st.cache_data
def get_coordinates(query):
    geolocator = Nominatim(user_agent="neurax_traffic_app")
    try:
        location = geolocator.geocode(query)
        if location:
            return [location.latitude, location.longitude]
    except:
        pass
    return [17.3850, 78.4867]


center_coords = get_coordinates(search_query)

# --- MAP RENDERING ---
m = folium.Map(location=center_coords, zoom_start=15,
               tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
               attr="Esri World Imagery")
Draw(export=False, position='topleft',
     draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False,
                   'rectangle': True}).add_to(m)

map_data = st_folium(m, width=1200, height=400)

if map_data and map_data.get("last_active_drawing"):
    geom = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    lons = [p[0] for p in geom]
    lats = [p[1] for p in geom]
    st.session_state.bbox = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"

# --- PIPELINE EXECUTION ---
if st.session_state.bbox:
    if st.button("⬇️ Fetch Image & Extract Road Graph", type="primary"):
        with st.spinner("Downloading high-res tile from ESRI..."):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={st.session_state.bbox}&bboxSR=4326&imageSR=4326&size=512,512&format=jpg&f=image"
            response = requests.get(url)
            if response.status_code == 200:
                satellite_img = Image.open(BytesIO(response.content))
                img_array = np.array(satellite_img)
            else:
                st.error("Failed to download image.")
                st.stop()

        with st.spinner("⚡ Processing AI & Math on RTX 4050..."):
            from pipeline import load_ai_model, process_and_heal_roads

            model = load_ai_model()

            binary_mask, graph, new_bridges, node_coords = process_and_heal_roads(
                img_array, model, healing_threshold, max_angle_error
            )

        st.success("✅ Network Extraction Complete!")

        # --- METRICS DISPLAY ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Extracted Nodes", len(graph.nodes))
        m2.metric("Total Routable Edges", len(graph.edges))
        m3.metric("Connected Sub-graphs", nx.number_connected_components(graph),
                  help="Lower is better. Represents isolated road networks.")

        st.markdown("---")

        # --- VISUALIZATION ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Phase I: AI Binary Mask")
            bw_mask = (binary_mask * 255).astype(np.uint8)
            st.image(bw_mask, use_container_width=True)

        with col2:
            st.subheader("Phase II: Topological NetworkX Graph")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(np.zeros((512, 512)), cmap='gray')  # Pure black background

            # Draw AI Found Roads (Green)
            for (s, e) in graph.edges():
                if 'pts' in graph[s][e]:
                    pts = graph[s][e]['pts']
                    ax.plot(pts[:, 1], pts[:, 0], 'g-', linewidth=2, alpha=0.7)

            # Draw Mathematically Healed Bridges (Cyan)
            for coord_a, coord_b in new_bridges:
                ax.plot([coord_a[1], coord_b[1]], [coord_a[0], coord_b[0]], 'c-', linewidth=3, label="Healed Edge")

            # Draw Intersections (Red dots)
            for node, coords in node_coords.items():
                ax.scatter(coords[1], coords[0], s=30, c='red', zorder=5)

            ax.axis('off')
            st.pyplot(fig)