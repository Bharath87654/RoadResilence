import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize, binary_closing, disk, dilation, remove_small_objects
from skimage.graph import route_through_array
import sknw
from scipy.spatial import distance, KDTree
import math
import networkx as nx

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ai_model(weights_path="weights/best_road_extractor.pth"):
    print(f"Loading AI Model onto {DEVICE}...")
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def process_and_heal_roads(image_array, model, ai_threshold=0.35, min_noise_size=30, healing_threshold=50.0,
                           max_angle_error=60.0):
    # --- PHASE I: AI INFERENCE ---
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor_img = transform(image=image_array)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(tensor_img)
        pred_mask_soft = torch.sigmoid(prediction).squeeze().cpu().numpy()

    binary_mask = (pred_mask_soft > ai_threshold).astype(np.uint8)
    binary_mask = dilation(binary_mask, disk(1))
    binary_mask = binary_closing(binary_mask, disk(5))
    binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=min_noise_size).astype(np.uint8)

    cost_map = np.where(pred_mask_soft > (ai_threshold * 0.5), 1.0 - pred_mask_soft + 0.01, 999.0)

    # --- PHASE II: GRAPH GENERATION ---
    skeleton = skeletonize(binary_mask).astype(np.uint16)
    graph = sknw.build_sknw(skeleton)

    # Deep copy for "Before/After" visualization
    graph_initial = graph.copy()

    dead_ends = [node for node, degree in dict(graph.degree()).items() if degree == 1]
    node_coords = {node: graph.nodes[node]['o'] for node in graph.nodes()}
    new_bridges_paths = []

    def get_road_direction(node, graph_ref, coords):
        neighbor = list(graph_ref.neighbors(node))[0]
        pts = graph_ref[node][neighbor].get('pts', [])
        step = min(5, max(1, len(pts) - 1))

        dist_to_start = distance.euclidean(coords[node], pts[0])
        dist_to_end = distance.euclidean(coords[node], pts[-1])

        p1, p2 = (pts[0], pts[step]) if dist_to_start < dist_to_end else (pts[-1], pts[-(step + 1)])
        dy, dx = p1[0] - p2[0], p1[1] - p2[1]
        length = math.hypot(dx, dy)
        return (dx / length, dy / length) if length > 0 else (0, 0)

    # --- PHASE III: GRAPH HEALING (Topological Mutation) ---
    if dead_ends:
        tree_endpoints = KDTree([node_coords[n] for n in dead_ends])
        connected_this_round = set()

        for i, node_a in enumerate(dead_ends):
            if node_a in connected_this_round: continue
            coord_a = node_coords[node_a]
            dir_a = get_road_direction(node_a, graph, node_coords)

            neighbors = tree_endpoints.query_ball_point(coord_a, healing_threshold)
            best_path, best_cost, best_target = None, float('inf'), None

            for j in neighbors:
                node_b = dead_ends[j]
                if node_a == node_b or node_b in connected_this_round or graph.has_edge(node_a, node_b):
                    continue

                # Check Component Isolation - Only heal across disjointed components!
                if nx.has_path(graph, node_a, node_b): continue

                coord_b = node_coords[node_b]
                dir_b = get_road_direction(node_b, graph, node_coords)

                bridge_dx, bridge_dy = coord_b[1] - coord_a[1], coord_b[0] - coord_a[0]
                euclidean_len = math.hypot(bridge_dx, bridge_dy)

                if euclidean_len > 0:
                    vec_ab = (bridge_dx / euclidean_len, bridge_dy / euclidean_len)
                    vec_ba = (-bridge_dx / euclidean_len, -bridge_dy / euclidean_len)
                    dot_a = (dir_a[0] * vec_ab[0]) + (dir_a[1] * vec_ab[1])
                    dot_b = (dir_b[0] * vec_ba[0]) + (dir_b[1] * vec_ba[1])

                    angle_a = math.degrees(math.acos(max(-1.0, min(1.0, dot_a))))
                    angle_b = math.degrees(math.acos(max(-1.0, min(1.0, dot_b))))

                    if angle_a < max_angle_error and angle_b < max_angle_error:
                        start_idx = (int(coord_a[0]), int(coord_a[1]))
                        end_idx = (int(coord_b[0]), int(coord_b[1]))

                        try:
                            indices, cost = route_through_array(cost_map, start_idx, end_idx, fully_connected=True)
                            if cost < 600.0 and cost < best_cost:
                                best_cost = cost
                                best_path = indices
                                best_target = node_b
                        except ValueError:
                            pass

            if best_path:
                # Add the logical edge directly to NetworkX
                graph.add_edge(node_a, best_target, weight=len(best_path), pts=np.array(best_path))
                new_bridges_paths.append(best_path)
                connected_this_round.add(node_a)
                connected_this_round.add(best_target)

    return pred_mask_soft, binary_mask, skeleton, graph_initial, graph, node_coords


def compute_isro_centrality(graph):
    """Calculates Betweenness and k-Core on the Largest Connected Component."""
    if len(graph.nodes) == 0: return None

    # Isolate the largest connected component for valid routing flow
    largest_cc = max(nx.connected_components(graph), key=len)
    lcc = graph.subgraph(largest_cc).copy()

    # 1. Betweenness Centrality (Critical Chokepoints)
    # Note: We use inverse weight because sknw edge weights are usually length (lower is faster)
    betweenness = nx.betweenness_centrality(lcc, weight='weight')

    # 2. k-Core (Network Resilience / Core dense areas)
    # Remove self-loops first as k-core doesn't support them
    lcc.remove_edges_from(nx.selfloop_edges(lcc))
    k_core = nx.core_number(lcc)

    nx.set_node_attributes(lcc, betweenness, 'betweenness')
    nx.set_node_attributes(lcc, k_core, 'k_core')

    return lcc