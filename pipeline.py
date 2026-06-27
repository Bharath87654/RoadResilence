import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize, binary_closing, disk, dilation, remove_small_objects
from skimage.graph import route_through_array
import sknw
from scipy.spatial import distance
import math
import networkx as nx
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
#  PHASE I  ·  Model loading & AI inference
# ─────────────────────────────────────────────

def load_ai_model(weights_path="weights/best_road_extractor.pth"):
    """Load UNet++ model. Decorated with @st.cache_resource in app.py."""
    resolved = Path(__file__).parent / weights_path
    print(f"Loading AI model onto {DEVICE} from {resolved}...")
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(resolved, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(image_array, model, ai_threshold=0.35, min_noise_size=30):
    """
    Phase I: AI inference → soft probability → binary mask → cost map.

    Returns
    -------
    pred_mask_soft : (H, W) float32  – raw sigmoid output
    binary_mask    : (H, W) uint8    – thresholded + cleaned mask
    cost_map       : (H, W) float64  – A* cost surface (low = road-likely)
    """
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor_img = transform(image=image_array)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(tensor_img)
        pred_mask_soft = torch.sigmoid(prediction).squeeze().cpu().numpy()

    # Clean up mask
    binary_mask = (pred_mask_soft > ai_threshold).astype(np.uint8)
    binary_mask = dilation(binary_mask, disk(1))
    binary_mask = binary_closing(binary_mask, disk(5))
    binary_mask = remove_small_objects(
        binary_mask.astype(bool), min_size=min_noise_size
    ).astype(np.uint8)

    # Cost surface: road-likely pixels are cheap; elsewhere is expensive
    # A* will prefer paths that stay on high-confidence areas
    cost_map = np.where(
        pred_mask_soft > (ai_threshold * 0.5),
        1.0 - pred_mask_soft + 0.01,   # 0.01 → 0.65  (cheap inside roads)
        999.0,                          # very expensive off-road
    )

    return pred_mask_soft, binary_mask, cost_map


# ─────────────────────────────────────────────
#  PHASE II  ·  Skeleton → Graph
# ─────────────────────────────────────────────

def build_graph_from_mask(binary_mask):
    """Skeletonise binary mask and build a NetworkX graph via sknw."""
    skeleton = skeletonize(binary_mask).astype(np.uint16)
    graph = sknw.build_sknw(skeleton)
    return skeleton, graph


def _get_node_coords(graph):
    """Return {node_id: (row, col)} for every node in the graph."""
    return {n: graph.nodes[n]["o"] for n in graph.nodes()}


# ─────────────────────────────────────────────
#  PHASE II  ·  Component-based graph healing
# ─────────────────────────────────────────────
#
#  Architecture (replaces the old KDTree / endpoint approach):
#
#  components = nx.connected_components(graph)
#      ↓
#  For each component → extract endpoints + centroid + bounding box
#      ↓
#  For every COMPONENT PAIR → find their mutually closest endpoints
#      → ONE candidate bridge per pair (not 400)
#      ↓
#  Score the bridge:   0.35·dist  +  0.30·avg_prob  +  0.20·direction  +  0.15·curvature
#      ↓
#  Accept / reject via hard gates (max_dist, min_prob, max_bend)
#      ↓
#  A* once per accepted bridge  →  graph.add_edge()
#      ↓
#  Repeat until no more bridges accepted (convergence)
#
# ─────────────────────────────────────────────

def _component_endpoints(graph, component_nodes):
    """Return degree-1 nodes (dead ends) within a component."""
    return [n for n in component_nodes if graph.degree(n) == 1]


def _component_centroid(graph, component_nodes):
    coords = np.array([graph.nodes[n]["o"] for n in component_nodes])
    return coords.mean(axis=0)


def _road_direction(graph, endpoint_node):
    """
    Estimate the direction vector at a dead-end node.
    Walks a few steps along its single edge to get a stable direction.
    Returns a unit (dy, dx) vector.
    """
    neighbors = list(graph.neighbors(endpoint_node))
    if not neighbors:
        return (0.0, 0.0)
    neighbor = neighbors[0]
    pts = graph[endpoint_node][neighbor].get("pts", [])
    if len(pts) < 2:
        return (0.0, 0.0)

    node_coord = graph.nodes[endpoint_node]["o"]
    step = min(5, len(pts) - 1)
    # Walk away from the endpoint
    d0 = distance.euclidean(node_coord, pts[0])
    d1 = distance.euclidean(node_coord, pts[-1])
    p1, p2 = (pts[0], pts[step]) if d0 < d1 else (pts[-1], pts[-(step + 1)])
    dy, dx = float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])
    length = math.hypot(dx, dy)
    return (dy / length, dx / length) if length > 0 else (0.0, 0.0)


def _bridge_score(coord_a, coord_b, dir_a, dir_b, cost_map,
                  w_dist=0.35, w_prob=0.30, w_dir=0.20, w_curv=0.15):
    """
    Composite bridge score (lower = better).
    Returns (score, path_indices) or (inf, None) if the bridge is invalid.
    Hard gates applied here before A* to avoid expensive calls.
    """
    dy = float(coord_b[0]) - float(coord_a[0])
    dx = float(coord_b[1]) - float(coord_a[1])
    euclidean_len = math.hypot(dx, dy)

    if euclidean_len == 0:
        return float("inf"), None

    # ── Hard gate 1: maximum bridge length ──────────────────────────────────
    MAX_BRIDGE_PX = 80
    if euclidean_len > MAX_BRIDGE_PX:
        return float("inf"), None

    # ── Direction alignment ──────────────────────────────────────────────────
    vec_ab = (dy / euclidean_len, dx / euclidean_len)
    vec_ba = (-dy / euclidean_len, -dx / euclidean_len)

    dot_a = max(-1.0, min(1.0, dir_a[0] * vec_ab[0] + dir_a[1] * vec_ab[1]))
    dot_b = max(-1.0, min(1.0, dir_b[0] * vec_ba[0] + dir_b[1] * vec_ba[1]))
    angle_a = math.degrees(math.acos(dot_a))
    angle_b = math.degrees(math.acos(dot_b))

    # ── Hard gate 2: direction must be reasonable ────────────────────────────
    MAX_ANGLE_DEG = 60
    if angle_a > MAX_ANGLE_DEG or angle_b > MAX_ANGLE_DEG:
        return float("inf"), None

    # ── Run A* on the cost surface ───────────────────────────────────────────
    start = (int(coord_a[0]), int(coord_a[1]))
    end   = (int(coord_b[0]), int(coord_b[1]))
    try:
        indices, a_star_cost = route_through_array(
            cost_map, start, end, fully_connected=True
        )
    except (ValueError, IndexError):
        return float("inf"), None

    if len(indices) == 0:
        return float("inf"), None

    # ── Hard gate 3: average probability must be reasonable ─────────────────
    MIN_AVG_PROB = 0.4
    path_arr = np.array(indices)
    rows = np.clip(path_arr[:, 0], 0, cost_map.shape[0] - 1)
    cols = np.clip(path_arr[:, 1], 0, cost_map.shape[1] - 1)
    # cost = 1 - prob + 0.01, so prob = 1 - cost + 0.01
    avg_cost   = cost_map[rows, cols].mean()
    avg_prob   = 1.0 - avg_cost + 0.01
    if avg_prob < MIN_AVG_PROB:
        return float("inf"), None

    # ── Hard gate 4: path curvature (bending) ───────────────────────────────
    if len(indices) >= 3:
        mid_idx = len(indices) // 2
        mid_pt  = np.array(indices[mid_idx])
        ideal   = (np.array(start) + np.array(end)) / 2
        bend_px = np.linalg.norm(mid_pt - ideal)
        MAX_BEND_PX = 30
        if bend_px > MAX_BEND_PX:
            return float("inf"), None
        curvature_score = bend_px / MAX_BEND_PX
    else:
        curvature_score = 0.0

    # ── Composite score ──────────────────────────────────────────────────────
    dist_score  = euclidean_len / MAX_BRIDGE_PX
    prob_score  = 1.0 - avg_prob          # lower prob = worse
    dir_score   = ((angle_a + angle_b) / 2) / MAX_ANGLE_DEG

    score = (w_dist  * dist_score  +
             w_prob  * prob_score  +
             w_dir   * dir_score   +
             w_curv  * curvature_score)

    return score, indices


def heal_graph_components(graph, cost_map, max_iterations=20):
    """
    Component-based graph healing.

    Each iteration:
      1. Find all connected components.
      2. For every PAIR of components, find their mutually closest endpoints.
      3. Score that single candidate bridge.
      4. Accept the best valid bridge across all pairs.
      5. Merge components via graph.add_edge().
    Repeat until no valid bridge found or max_iterations reached.

    Returns the mutated graph and a list of bridge paths (for visualisation).
    """
    bridge_paths = []

    for iteration in range(max_iterations):
        components = list(nx.connected_components(graph))
        if len(components) <= 1:
            print(f"  Fully connected after {iteration} iterations.")
            break

        # Build per-component endpoint lists
        comp_endpoints = {}
        for i, comp in enumerate(components):
            eps = _component_endpoints(graph, comp)
            if eps:
                comp_endpoints[i] = eps

        comp_ids = list(comp_endpoints.keys())
        if len(comp_ids) < 2:
            break

        best_score  = float("inf")
        best_bridge = None   # (node_a, node_b, path_indices)

        # Compare every pair of components
        for i in range(len(comp_ids)):
            for j in range(i + 1, len(comp_ids)):
                ci, cj = comp_ids[i], comp_ids[j]
                eps_i  = comp_endpoints[ci]
                eps_j  = comp_endpoints[cj]

                # ── Find the single closest endpoint pair ────────────────────
                best_pair_dist = float("inf")
                closest_pair   = None
                for na in eps_i:
                    coord_a = graph.nodes[na]["o"]
                    for nb in eps_j:
                        coord_b = graph.nodes[nb]["o"]
                        d = distance.euclidean(
                                coord_a.astype(float), coord_b.astype(float)
                            )
                        if d < best_pair_dist:
                            best_pair_dist = d
                            closest_pair   = (na, nb)

                if closest_pair is None:
                    continue

                na, nb     = closest_pair
                coord_a    = graph.nodes[na]["o"]
                coord_b    = graph.nodes[nb]["o"]
                dir_a      = _road_direction(graph, na)
                dir_b      = _road_direction(graph, nb)

                score, path = _bridge_score(
                    coord_a, coord_b, dir_a, dir_b, cost_map
                )

                if score < best_score:
                    best_score  = score
                    best_bridge = (na, nb, path)

        if best_bridge is None:
            print(f"  No valid bridge found at iteration {iteration}. Stopping.")
            break

        # ── Accept the best bridge this iteration ───────────────────────────
        na, nb, path_indices = best_bridge
        graph.add_edge(
            na, nb,
            weight  = len(path_indices),
            pts     = np.array(path_indices),
            healed  = True,          # flag for visualisation
        )
        bridge_paths.append(path_indices)
        print(f"  Iter {iteration}: merged components via nodes {na}↔{nb}, "
              f"score={best_score:.3f}, path_len={len(path_indices)}")

    return graph, bridge_paths


# ─────────────────────────────────────────────
#  PHASE II  ·  Enriched edge attributes
# ─────────────────────────────────────────────

def enrich_edge_attributes(graph, pred_mask_soft):
    """
    Store edge-level metadata on every edge:
      - length   : number of pixels in the path
      - avg_conf : average UNet++ confidence along the edge
      - healed   : True if added by the healing step
    """
    H, W = pred_mask_soft.shape
    for u, v, data in graph.edges(data=True):
        pts = data.get("pts", None)
        if pts is not None and len(pts) > 0:
            pts_arr = np.array(pts)
            rows = np.clip(pts_arr[:, 0].astype(int), 0, H - 1)
            cols = np.clip(pts_arr[:, 1].astype(int), 0, W - 1)
            data["length"]   = len(pts)
            data["avg_conf"] = float(pred_mask_soft[rows, cols].mean())
        else:
            data["length"]   = data.get("weight", 1)
            data["avg_conf"] = 0.0
        data.setdefault("healed", False)
    return graph


# ─────────────────────────────────────────────
#  PHASE III  ·  Centrality & resilience
# ─────────────────────────────────────────────

def compute_isro_centrality(graph):
    """
    Calculates on the Largest Connected Component:
      - Node Betweenness Centrality   → critical intersections
      - Edge Betweenness Centrality   → critical arterial roads
      - k-Core decomposition          → network resilience zones
      - Global Efficiency             → baseline resilience metric
    """
    if len(graph.nodes) == 0:
        return None

    # Isolate LCC
    largest_cc = max(nx.connected_components(graph), key=len)
    lcc = graph.subgraph(largest_cc).copy()

    # Remove self-loops (k-core requirement)
    lcc.remove_edges_from(nx.selfloop_edges(lcc))

    # 1. Node betweenness (use edge length as weight → shorter = faster)
    node_betweenness = nx.betweenness_centrality(lcc, weight="length")

    # 2. Edge betweenness (road-level criticality)
    edge_betweenness = nx.edge_betweenness_centrality(lcc, weight="length")

    # 3. k-Core
    k_core = nx.core_number(lcc)

    # Attach to graph
    nx.set_node_attributes(lcc, node_betweenness, "betweenness")
    nx.set_node_attributes(lcc, k_core,           "k_core")
    nx.set_edge_attributes(lcc, edge_betweenness, "edge_betweenness")

    # 4. Baseline global efficiency
    lcc.graph["baseline_efficiency"] = nx.global_efficiency(lcc)

    return lcc


def run_disaster_simulation(lcc_graph, n_failed=3):
    """
    Remove the top-n highest-betweenness nodes (simulated failures).

    Returns
    -------
    sim_graph         : NetworkX graph after removal
    failed_nodes      : list of removed node IDs
    baseline_eff      : float
    post_eff          : float
    efficiency_drop_pct : float
    resilience_index  : float  (baseline / post, lower = more vulnerable)
    """
    baseline_eff = lcc_graph.graph.get("baseline_efficiency",
                                       nx.global_efficiency(lcc_graph))

    sorted_nodes = sorted(
        lcc_graph.nodes(data=True),
        key=lambda x: x[1].get("betweenness", 0),
        reverse=True,
    )
    failed_nodes = [n[0] for n in sorted_nodes[:n_failed]]

    sim_graph = lcc_graph.copy()
    sim_graph.remove_nodes_from(failed_nodes)

    post_eff = nx.global_efficiency(sim_graph)

    efficiency_drop_pct = (
        ((baseline_eff - post_eff) / baseline_eff) * 100
        if baseline_eff > 0 else 0.0
    )
    resilience_index = (baseline_eff / post_eff) if post_eff > 0 else float("inf")

    return sim_graph, failed_nodes, baseline_eff, post_eff, efficiency_drop_pct, resilience_index


# ─────────────────────────────────────────────
#  Public entry-point (called from app.py)
# ─────────────────────────────────────────────

def process_and_heal_roads(image_array, model,
                           ai_threshold=0.35,
                           min_noise_size=30,
                           healing_threshold=80.0,   # kept for UI slider compat.
                           max_angle_error=60.0):    # kept for UI slider compat.
    """
    Full pipeline: image → segmentation → skeleton → graph → healed graph.

    Returns
    -------
    pred_mask_soft  : (H, W) float32
    binary_mask     : (H, W) uint8
    skeleton        : (H, W) uint16
    graph_initial   : NetworkX graph before healing
    graph_healed    : NetworkX graph after healing
    node_coords     : {node_id: (row, col)}
    bridge_paths    : list of pixel-path arrays for the healed bridges
    """
    # Phase I
    pred_mask_soft, binary_mask, cost_map = run_inference(
        image_array, model, ai_threshold, min_noise_size
    )

    # Phase II-a: build graph
    skeleton, graph = build_graph_from_mask(binary_mask)
    graph_initial   = graph.copy()

    # Phase II-b: component-based healing
    graph, bridge_paths = heal_graph_components(graph, cost_map, max_iterations=30)

    # Phase II-c: enrich edge attributes
    graph = enrich_edge_attributes(graph, pred_mask_soft)

    node_coords = _get_node_coords(graph)
    return (pred_mask_soft, binary_mask, skeleton,
            graph_initial, graph, node_coords, bridge_paths)