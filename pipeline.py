import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize, binary_closing, disk, dilation
import sknw
from scipy.spatial import distance, KDTree
import math
import networkx as nx

# Automatically route to your RTX 4050
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ai_model(weights_path="weights/best_road_extractor.pth"):
    """Loads the 40-epoch U-Net++ brain directly into the GPU."""
    print(f"Loading AI Model onto {DEVICE}...")
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def process_and_heal_roads(image_array, model, healing_threshold=41.0, max_angle_error=45.0):
    """
    Phase I: GPU Inference + Morphological Noise Cleaning
    Phase II: KDTree Vector Healing with Hard Graph Mutation
    """
    # --- PHASE I: AI INFERENCE ---
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor_img = transform(image=image_array)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(tensor_img)
        pred_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()

    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # --- NEW: MORPHOLOGICAL PRE-PROCESSING ---
    # 1. Dilation: Expands road pixels slightly to jump tiny 1-2 pixel micro-gaps
    binary_mask = dilation(binary_mask, disk(1))
    # 2. Closing: Fills in holes and smoothes jagged edges caused by shadows
    binary_mask = binary_closing(binary_mask, disk(3))

    # --- PHASE II: TOPOLOGICAL HEALING ---
    skeleton = skeletonize(binary_mask).astype(np.uint16)
    graph = sknw.build_sknw(skeleton)

    dead_ends = [node for node, degree in dict(graph.degree()).items() if degree == 1]
    node_coords = {node: graph.nodes[node]['o'] for node in graph.nodes()}
    new_bridges = []

    def get_road_direction(node, graph, coords):
        neighbor = list(graph.neighbors(node))[0]
        pts = graph[node][neighbor].get('pts', [])
        step = min(5, max(1, len(pts) - 1))

        dist_to_start = distance.euclidean(coords[node], pts[0])
        dist_to_end = distance.euclidean(coords[node], pts[-1])

        if dist_to_start < dist_to_end:
            p1, p2 = pts[0], pts[step]
        else:
            p1, p2 = pts[-1], pts[-(step + 1)]

        dy = p1[0] - p2[0]
        dx = p1[1] - p2[1]
        length = math.hypot(dx, dy)
        if length == 0: return 0, 0
        return dx / length, dy / length

    # --- BUILD KDTREES ---
    if not dead_ends:
        return binary_mask, graph, new_bridges, node_coords

    end_coords_list = [node_coords[n] for n in dead_ends]
    tree_endpoints = KDTree(end_coords_list)

    edge_pixels = []
    for u, v, data in graph.edges(data=True):
        pts = data.get('pts', [])
        for i in range(0, len(pts), 5):
            edge_pixels.append((pts[i][0], pts[i][1]))

    tree_edges = KDTree(edge_pixels) if edge_pixels else None

    # --- KDTREE SEARCH & GRAPH MUTATION ---
    connected_this_round = set()

    for i, node_a in enumerate(dead_ends):
        if node_a in connected_this_round: continue

        coord_a = node_coords[node_a]
        dir_a = get_road_direction(node_a, graph, node_coords)

        # STRATEGY A: Dead-End to Dead-End
        neighbors = tree_endpoints.query_ball_point(coord_a, healing_threshold)
        snapped = False

        for j in neighbors:
            node_b = dead_ends[j]
            if node_a == node_b or node_b in connected_this_round: continue
            if graph.has_edge(node_a, node_b): continue

            coord_b = node_coords[node_b]
            dir_b = get_road_direction(node_b, graph, node_coords)

            bridge_dx, bridge_dy = coord_b[1] - coord_a[1], coord_b[0] - coord_a[0]
            bridge_len = math.hypot(bridge_dx, bridge_dy)

            if bridge_len > 0:
                vec_ab = (bridge_dx / bridge_len, bridge_dy / bridge_len)
                vec_ba = (-bridge_dx / bridge_len, -bridge_dy / bridge_len)

                dot_a = (dir_a[0] * vec_ab[0]) + (dir_a[1] * vec_ab[1])
                dot_b = (dir_b[0] * vec_ba[0]) + (dir_b[1] * vec_ba[1])

                angle_a = math.degrees(math.acos(max(-1.0, min(1.0, dot_a))))
                angle_b = math.degrees(math.acos(max(-1.0, min(1.0, dot_b))))

                if angle_a < max_angle_error and angle_b < max_angle_error:
                    graph.add_edge(node_a, node_b, weight=bridge_len)
                    new_bridges.append((coord_a, coord_b))
                    connected_this_round.add(node_a)
                    connected_this_round.add(node_b)
                    snapped = True
                    break

        # STRATEGY B: Mid-Road Snapping (HARD GRAPH MUTATION)
        if not snapped and tree_edges is not None:
            dist, idx = tree_edges.query(coord_a, distance_upper_bound=healing_threshold)

            if dist != float('inf'):
                target_pixel = edge_pixels[idx]

                bridge_dx = target_pixel[1] - coord_a[1]
                bridge_dy = target_pixel[0] - coord_a[0]
                bridge_len = math.hypot(bridge_dx, bridge_dy)

                if bridge_len > 0:
                    vec_ab = (bridge_dx / bridge_len, bridge_dy / bridge_len)
                    dot_a = (dir_a[0] * vec_ab[0]) + (dir_a[1] * vec_ab[1])
                    angle_a = math.degrees(math.acos(max(-1.0, min(1.0, dot_a))))

                    if angle_a < max_angle_error:
                        # ⚠️ The Graph Mutation: Inject a new routable node mid-road!
                        new_node_id = max(graph.nodes()) + 1
                        graph.add_node(new_node_id, o=np.array(target_pixel))
                        graph.add_edge(node_a, new_node_id, weight=bridge_len)

                        # Add to node_coords so UI can draw it
                        node_coords[new_node_id] = np.array(target_pixel)

                        new_bridges.append((coord_a, target_pixel))
                        connected_this_round.add(node_a)

    return binary_mask, graph, new_bridges, node_coords