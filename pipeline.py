import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize
import sknw
from scipy.spatial import distance
import math

# Automatically route to your RTX 4050
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ai_model(weights_path="weights/best_road_extractor.pth"):
    """Loads the 40-epoch U-Net++ brain directly into the GPU."""
    print(f"Loading AI Model onto {DEVICE}...")
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)

    # Safely load weights and push the model to the GPU
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def process_and_heal_roads(image_array, model, healing_threshold=20.0, max_angle_error=30.0):
    """
    Phase I: GPU Inference (Extracts roads)
    Phase II: CPU Math (Finds dead ends and builds bridges based on strict vector alignment)
    """

    # --- PHASE I: AI INFERENCE (GPU) ---
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

    # --- PHASE II: TOPOLOGICAL HEALING (CPU) ---
    # Shrink to 1-pixel wide skeleton and build graph
    skeleton = skeletonize(binary_mask).astype(np.uint16)
    graph = sknw.build_sknw(skeleton)

    dead_ends = [node for node, degree in dict(graph.degree()).items() if degree == 1]
    node_coords = {node: graph.nodes[node]['o'] for node in graph.nodes()}
    new_bridges = []

    def get_road_direction(node, graph, coords):
        """Calculates the orientation vector of the road leading to the endpoint."""
        neighbor = list(graph.neighbors(node))[0]
        dy = coords[node][0] - coords[neighbor][0]
        dx = coords[node][1] - coords[neighbor][1]
        length = math.hypot(dx, dy)
        if length == 0: return 0, 0
        return dx / length, dy / length

    # Nearest Neighbor Search & Gap Connection
    for i in range(len(dead_ends)):
        for j in range(i + 1, len(dead_ends)):
            node_a = dead_ends[i]
            node_b = dead_ends[j]

            coord_a = node_coords[node_a]
            coord_b = node_coords[node_b]

            # 1. Compute Distance
            dist = distance.euclidean(coord_a, coord_b)

            if dist < healing_threshold:
                # 2. Compute Direction/Orientation
                dir_a = get_road_direction(node_a, graph, node_coords)
                dir_b = get_road_direction(node_b, graph, node_coords)

                bridge_dx = coord_b[1] - coord_a[1]
                bridge_dy = coord_b[0] - coord_a[0]
                bridge_len = math.hypot(bridge_dx, bridge_dy)

                if bridge_len > 0:
                    vec_ab = (bridge_dx / bridge_len, bridge_dy / bridge_len)
                    vec_ba = (-bridge_dx / bridge_len, -bridge_dy / bridge_len)

                    dot_a = (dir_a[0] * vec_ab[0]) + (dir_a[1] * vec_ab[1])
                    dot_b = (dir_b[0] * vec_ba[0]) + (dir_b[1] * vec_ba[1])

                    # 3. Strict Angle Alignment Check
                    angle_a = math.degrees(math.acos(max(-1.0, min(1.0, dot_a))))
                    angle_b = math.degrees(math.acos(max(-1.0, min(1.0, dot_b))))

                    if angle_a < max_angle_error and angle_b < max_angle_error:
                        if not graph.has_edge(node_a, node_b):
                            graph.add_edge(node_a, node_b, weight=dist)
                            new_bridges.append((coord_a, coord_b))

    return binary_mask, graph, new_bridges, node_coords