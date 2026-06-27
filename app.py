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
import matplotlib.cm as cm
import networkx as nx

st.set_page_config(layout="wide", page_title="Urban Resilience Analysis")

st.title("🚦 ISRO Topological Graph Intelligence Pipeline")
st.markdown("Transforming satellite occlusion into a mathematically healed, centrality-analyzed routing graph.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.header("🌍 Map Navigation")
search_query = st.sidebar.text_input("Search for a City:", "Hyderabad, India")

st.sidebar.markdown("---")
st.sidebar.header("🧠 Phase I Controls")
ai_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 0.9, 0.35, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Phase II Controls")
healing_threshold = st.sidebar.slider("Max Bridge Length (px)", 10.0, 120.0, 80.0, step=5.0)
max_angle_error   = st.sidebar.slider("Max Angle Deviation (°)", 10.0, 90.0, 60.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.header("💥 Phase V Controls")
n_failed_nodes = st.sidebar.slider("Nodes to Fail (Simulation)", 1, 10, 3, step=1)

if "bbox" not in st.session_state:
    st.session_state.bbox = None

# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_data
def get_coordinates(query):
    geolocator = Nominatim(user_agent="isro_pipeline_app")
    try:
        location = geolocator.geocode(query)
        if location:
            return [location.latitude, location.longitude]
    except Exception:
        pass
    return [17.3850, 78.4867]


@st.cache_resource
def get_model():
    from pipeline import load_ai_model
    return load_ai_model()


# ─── Map UI ──────────────────────────────────────────────────────────────────

m = folium.Map(
    location=get_coordinates(search_query),
    zoom_start=15,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
)
Draw(
    export=False,
    position="topleft",
    draw_options={
        "polyline": False, "polygon": False, "circle": False,
        "marker": False, "circlemarker": False, "rectangle": True,
    },
).add_to(m)
map_data = st_folium(m, width=1200, height=400)

if map_data and map_data.get("last_active_drawing"):
    geom = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    st.session_state.bbox = (
        f"{min(p[0] for p in geom)},{min(p[1] for p in geom)},"
        f"{max(p[0] for p in geom)},{max(p[1] for p in geom)}"
    )

# ─── Pipeline trigger ────────────────────────────────────────────────────────

if st.session_state.bbox:
    if st.button("⬇️ Run Full ISRO Pipeline", type="primary"):

        with st.spinner("Downloading satellite tile..."):
            url = (
                f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Imagery/MapServer/export?bbox={st.session_state.bbox}"
                f"&bboxSR=4326&imageSR=4326&size=512,512&format=jpg&f=image"
            )
            response = requests.get(url)
            if response.status_code != 200:
                st.error("Failed to download satellite tile.")
                st.stop()
            img_array = np.array(Image.open(BytesIO(response.content)))

        with st.spinner("⚡ Running segmentation + component healing..."):
            from pipeline import process_and_heal_roads, compute_isro_centrality, run_disaster_simulation

            model = get_model()
            (pred_mask_soft, binary_mask, skeleton,
             graph_initial, graph_healed, node_coords,
             bridge_paths) = process_and_heal_roads(
                img_array, model, ai_threshold, 30,
                healing_threshold, max_angle_error,
            )

        with st.spinner("Computing centrality metrics..."):
            lcc_graph = compute_isro_centrality(graph_healed)

        st.success("✅ Graph Analysis Complete!")

        # ─── Tabs ─────────────────────────────────────────────────────────────

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🛰️ 1. Segmentation",
            "🕸️ 2. Skeleton",
            "🩹 3. Graph Healing",
            "🔥 4. Centrality Analysis",
            "💥 5. Simulation",
        ])

        # ── Tab 1: Segmentation ───────────────────────────────────────────────
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Raw Satellite Input")
                st.image(img_array, use_container_width=True)
            with col2:
                st.subheader("Soft Probability Map")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(pred_mask_soft, cmap="magma")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

        # ── Tab 2: Skeleton ───────────────────────────────────────────────────
        with tab2:
            st.subheader("1D Topological Skeleton")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(skeleton, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        # ── Tab 3: Graph Healing ──────────────────────────────────────────────
        with tab3:
            st.subheader("Component-Based Graph Healing")

            n_before = nx.number_connected_components(graph_initial)
            n_after  = nx.number_connected_components(graph_healed)
            lcc_size_before = len(max(nx.connected_components(graph_initial), key=len))
            lcc_size_after  = len(max(nx.connected_components(graph_healed),  key=len))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Components Before", n_before)
            m2.metric("Components After",  n_after, delta=f"-{n_before - n_after}")
            m3.metric("LCC Size Before", lcc_size_before)
            m4.metric("LCC Size After",  lcc_size_after,
                      delta=f"+{lcc_size_after - lcc_size_before}")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("**Initial Graph (Broken)**")
                fig_init, ax_init = plt.subplots(figsize=(8, 8))
                ax_init.imshow(img_array, alpha=0.5)
                for (s, e) in graph_initial.edges():
                    pts = graph_initial[s][e].get("pts")
                    if pts is not None:
                        ax_init.plot(pts[:, 1], pts[:, 0], "r-", linewidth=1.5)
                ax_init.axis("off")
                st.pyplot(fig_init)
                plt.close(fig_init)

            with colB:
                st.markdown("**Healed Graph — green = original, cyan = new bridge**")
                fig_heal, ax_heal = plt.subplots(figsize=(8, 8))
                ax_heal.imshow(img_array, alpha=0.5)
                for (s, e, data) in graph_healed.edges(data=True):
                    pts = data.get("pts")
                    if pts is not None:
                        # NEW bridge edges are tagged healed=True in pipeline
                        if data.get("healed", False):
                            ax_heal.plot(pts[:, 1], pts[:, 0], "c--", linewidth=2.5)
                        else:
                            ax_heal.plot(pts[:, 1], pts[:, 0], "g-",  linewidth=1.5)
                ax_heal.axis("off")
                st.pyplot(fig_heal)
                plt.close(fig_heal)

        # ── Tab 4: Centrality Analysis ────────────────────────────────────────
        with tab4:
            st.subheader("Network Criticality — Gatekeeper Nodes & Arterial Roads")

            if lcc_graph:
                # Sub-tabs for node vs edge centrality
                sub1, sub2 = st.tabs(["Node Betweenness", "Edge Criticality & k-Core"])

                with sub1:
                    fig_cent, ax_cent = plt.subplots(figsize=(10, 10))
                    ax_cent.imshow(img_array, alpha=0.35)

                    # Draw edges coloured by edge betweenness
                    eb_vals = [
                        d.get("edge_betweenness", 0.0)
                        for _, _, d in lcc_graph.edges(data=True)
                    ]
                    eb_max = max(eb_vals) if eb_vals else 1.0
                    edge_cmap = cm.get_cmap("YlOrRd")

                    for (s, e, data) in lcc_graph.edges(data=True):
                        pts = data.get("pts")
                        if pts is not None:
                            eb_norm = data.get("edge_betweenness", 0.0) / (eb_max + 1e-9)
                            color   = edge_cmap(eb_norm)
                            ax_cent.plot(pts[:, 1], pts[:, 0],
                                         color=color, linewidth=1.5, alpha=0.7)

                    # Draw nodes scaled by betweenness
                    b_vals = [d.get("betweenness", 0.0)
                              for _, d in lcc_graph.nodes(data=True)]
                    b_max  = max(b_vals) if b_vals else 1.0
                    node_cmap = cm.get_cmap("autumn_r")

                    x_list, y_list, c_list, s_list = [], [], [], []
                    for n, data in lcc_graph.nodes(data=True):
                        coord = node_coords[n]
                        b     = data.get("betweenness", 0.0)
                        x_list.append(coord[1])
                        y_list.append(coord[0])
                        c_list.append(b)
                        s_list.append(60 + 400 * (b / (b_max + 1e-9)))  # size ∝ betweenness

                    sc = ax_cent.scatter(x_list, y_list, c=c_list,
                                        s=s_list, cmap="autumn_r",
                                        edgecolors="white", linewidths=0.5, zorder=5)
                    plt.colorbar(sc, ax=ax_cent, label="Node Betweenness Score")
                    ax_cent.axis("off")
                    st.pyplot(fig_cent)
                    plt.close(fig_cent)

                with sub2:
                    fig_kc, ax_kc = plt.subplots(figsize=(10, 10))
                    ax_kc.imshow(img_array, alpha=0.35)

                    # Draw nodes coloured by k-core level
                    k_vals = [d.get("k_core", 0) for _, d in lcc_graph.nodes(data=True)]
                    k_max  = max(k_vals) if k_vals else 1

                    xk, yk, ck = [], [], []
                    for n, data in lcc_graph.nodes(data=True):
                        coord = node_coords[n]
                        xk.append(coord[1])
                        yk.append(coord[0])
                        ck.append(data.get("k_core", 0))

                    sc2 = ax_kc.scatter(xk, yk, c=ck, s=80, cmap="viridis",
                                        edgecolors="white", linewidths=0.5, zorder=5)
                    plt.colorbar(sc2, ax=ax_kc, label="k-Core Level (higher = more resilient zone)")
                    ax_kc.axis("off")
                    st.pyplot(fig_kc)
                    plt.close(fig_kc)

            else:
                st.error("Graph is too fragmented to compute centrality.")

        # ── Tab 5: Disaster Simulation ────────────────────────────────────────
        with tab5:
            st.subheader("💥 Disaster Simulation: Cascading Network Failure")
            st.markdown(
                f"Simulating the removal of the top **{n_failed_nodes}** "
                "'Gatekeeper' intersections (e.g. severe flooding, structural collapse)."
            )

            if lcc_graph:
                (sim_graph, failed_nodes,
                 baseline_eff, post_eff,
                 efficiency_drop, resilience_index) = run_disaster_simulation(
                    lcc_graph, n_failed=n_failed_nodes
                )

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Baseline Efficiency",    f"{baseline_eff:.4f}")
                m2.metric("Post-Failure Efficiency", f"{post_eff:.4f}")
                m3.metric("Network Collapse",        f"{efficiency_drop:.1f}%",
                          delta="▼ Critical", delta_color="inverse")
                m4.metric("Resilience Index (R)",   f"{resilience_index:.3f}",
                          help="R = baseline/post. Higher = more vulnerable.")

                st.caption(
                    "Resilience Index R = E_baseline / E_post. "
                    "R close to 1.0 = robust. R >> 1.0 = fragile."
                )

                # Visualisation
                fig_sim, ax_sim = plt.subplots(figsize=(10, 10))
                ax_sim.imshow(img_array, alpha=0.3)

                # Surviving roads (grey)
                for (s, e, data) in sim_graph.edges(data=True):
                    pts = data.get("pts")
                    if pts is not None:
                        ax_sim.plot(pts[:, 1], pts[:, 0],
                                    color="gray", linewidth=1.2, alpha=0.6)

                # Failed nodes — large red X
                for failed_node in failed_nodes:
                    coord = node_coords[failed_node]
                    ax_sim.plot(coord[1], coord[0], "rx",
                                markersize=24, markeredgewidth=4,
                                label="Failed intersection")

                handles, labels = ax_sim.get_legend_handles_labels()
                if handles:
                    ax_sim.legend(handles[:1], labels[:1],
                                  loc="upper right", fontsize=10)
                ax_sim.axis("off")
                st.pyplot(fig_sim)
                plt.close(fig_sim)

                # Connectivity breakdown
                n_comps_post = nx.number_connected_components(sim_graph)
                st.info(
                    f"After removing {n_failed_nodes} nodes: "
                    f"**{n_comps_post}** disconnected component(s) remain. "
                    f"Efficiency dropped by **{efficiency_drop:.1f}%**."
                )
            else:
                st.error("Graph is too fragmented to run a simulation.")