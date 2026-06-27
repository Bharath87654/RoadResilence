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

st.set_page_config(layout="wide", page_title="Urban Resilience Analysis")

st.title("🚦 ISRO Topological Graph Intelligence Pipeline")
st.markdown("Transforming satellite occlusion into a mathematically healed, centrality-analyzed routing graph.")

st.sidebar.header("🌍 Map Navigation")
search_query = st.sidebar.text_input("Search for a City:", "Hyderabad, India")

st.sidebar.markdown("---")
st.sidebar.header("🧠 Phase I Controls")
ai_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 0.9, 0.35, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Phase II Controls")
healing_threshold = st.sidebar.slider("MAX_DISTANCE (px)", 10.0, 100.0, 50.0, step=5.0)
max_angle_error = st.sidebar.slider("ANGLE_THRESHOLD (°)", 10.0, 90.0, 60.0, step=5.0)

if "bbox" not in st.session_state:
    st.session_state.bbox = None


@st.cache_data
def get_coordinates(query):
    geolocator = Nominatim(user_agent="neurax_traffic_app")
    try:
        location = geolocator.geocode(query)
        if location: return [location.latitude, location.longitude]
    except:
        pass
    return [17.3850, 78.4867]


# MAP UI
m = folium.Map(location=get_coordinates(search_query), zoom_start=15,
               tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
               attr="Esri World Imagery")
Draw(export=False, position='topleft',
     draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False,
                   'rectangle': True}).add_to(m)
map_data = st_folium(m, width=1200, height=400)

if map_data and map_data.get("last_active_drawing"):
    geom = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    st.session_state.bbox = f"{min(p[0] for p in geom)},{min(p[1] for p in geom)},{max(p[0] for p in geom)},{max(p[1] for p in geom)}"

if st.session_state.bbox:
    if st.button("⬇️ Run Full ISRO Pipeline", type="primary"):
        with st.spinner("Downloading and parsing data..."):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={st.session_state.bbox}&bboxSR=4326&imageSR=4326&size=512,512&format=jpg&f=image"
            response = requests.get(url)
            img_array = np.array(Image.open(BytesIO(response.content))) if response.status_code == 200 else st.stop()

        with st.spinner("⚡ Executing Deep Learning & Graph Math..."):
            from pipeline import load_ai_model, process_and_heal_roads, compute_isro_centrality

            model = load_ai_model()

            # Execute the pipeline
            pred_mask_soft, binary_mask, skeleton, graph_initial, graph_healed, node_coords = process_and_heal_roads(
                img_array, model, ai_threshold, 30, healing_threshold, max_angle_error
            )

            # Compute ISRO Metrics
            lcc_graph = compute_isro_centrality(graph_healed)

        st.success("✅ Graph Analysis Complete!")

        # --- THE 5-TAB ARCHITECTURE ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🛰️ 1. Segmentation",
            "🕸️ 2. Skeleton",
            "🩹 3. Graph Healing",
            "🔥 4. Centrality Analysis",
            "💥 5. Simulation"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Raw Satellite Input")
                st.image(img_array, use_container_width=True)
            with col2:
                st.subheader("Soft Probability Map")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(pred_mask_soft, cmap='magma')
                ax.axis('off')
                st.pyplot(fig)

        with tab2:
            st.subheader("1D Topological Skeleton")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(skeleton, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

        with tab3:
            st.subheader("Component-Aware A* Healing")
            colA, colB = st.columns(2)

            with colA:
                st.markdown("**Initial Graph (Broken)**")
                fig_init, ax_init = plt.subplots(figsize=(8, 8))
                ax_init.imshow(img_array, alpha=0.5)
                for (s, e) in graph_initial.edges():
                    if 'pts' in graph_initial[s][e]:
                        pts = graph_initial[s][e]['pts']
                        ax_init.plot(pts[:, 1], pts[:, 0], 'r-', linewidth=2)
                ax_init.axis('off')
                st.pyplot(fig_init)

            with colB:
                st.markdown("**Healed Graph (Connected)**")
                fig_heal, ax_heal = plt.subplots(figsize=(8, 8))
                ax_heal.imshow(img_array, alpha=0.5)

                # Draw all edges in the healed graph
                for (s, e) in graph_healed.edges():
                    if 'pts' in graph_healed[s][e]:
                        pts = graph_healed[s][e]['pts']
                        # Differentiate new vs old visually (if it exists in initial, it's green, else cyan)
                        color = 'g-' if graph_initial.has_edge(s, e) else 'c--'
                        width = 2 if color == 'g-' else 3
                        ax_heal.plot(pts[:, 1], pts[:, 0], color, linewidth=width)

                ax_heal.axis('off')
                st.pyplot(fig_heal)

        with tab4:
            st.subheader("Betweenness Centrality (Gatekeeper Nodes)")
            if lcc_graph:
                fig_cent, ax_cent = plt.subplots(figsize=(10, 10))
                ax_cent.imshow(img_array, alpha=0.4)  # Dim background

                # Extract coordinates and centrality scores
                x, y, c_scores = [], [], []
                for n, data in lcc_graph.nodes(data=True):
                    coords = node_coords[n]
                    x.append(coords[1])
                    y.append(coords[0])
                    c_scores.append(data.get('betweenness', 0.0))

                # Scatter plot colored by Betweenness (Red = Critical Bottleneck)
                sc = ax_cent.scatter(x, y, c=c_scores, cmap='autumn_r', s=100, edgecolors='white', zorder=5)

                # Draw edges softly
                for (s, e) in lcc_graph.edges():
                    if 'pts' in lcc_graph[s][e]:
                        pts = lcc_graph[s][e]['pts']
                        ax_cent.plot(pts[:, 1], pts[:, 0], 'w-', linewidth=1, alpha=0.5)

                plt.colorbar(sc, ax=ax_cent, label="Betweenness Score")
                ax_cent.axis('off')
                st.pyplot(fig_cent)
            else:
                st.error("Graph is too fragmented to compute centrality.")

        with tab5:
            st.subheader("Disaster Simulation & Flow Interruption")
            st.info("Module active. Ready for Node-Removal mechanics.")