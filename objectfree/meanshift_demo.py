import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import MeanShift, estimate_bandwidth
from bbox_refinement import BBoxRefinement  # Import từ file hiện tại

# Import detector từ objectfree
from inference_dino import LoadDetector

def compute_entropy_map(region, window_size=20):
    """Tính entropy map như code bạn cung cấp"""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    entropy_map = np.zeros((h, w), dtype=np.float32)
    pad = window_size // 2
    gray_padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, 0)
    step = max(1, window_size // 4)
    for i in range(0, h, step):
        for j in range(0, w, step):
            window = gray_padded[i:i+window_size, j:j+window_size]
            hist, _ = np.histogram(window, bins=128, range=(0, 256))
            hist = hist / (window_size * window_size + 1e-10)
            entropy_val = -np.sum(hist * np.log2(hist + 1e-10))
            entropy_map[i:i+step, j:j+step] = entropy_val
    if step > 1:
        entropy_map = cv2.resize(entropy_map, (w, h), interpolation=cv2.INTER_LINEAR)
    return entropy_map

def threshold_entropy(entropy_map, method="mean"):
    """Threshold entropy để get mask high-entropy regions"""
    if method == "mean":
        th = np.mean(entropy_map)
    elif method == "median":
        th = np.median(entropy_map)
    else:
        th = np.quantile(entropy_map, 0.75)
    mask_high = (entropy_map >= th).astype(np.float32)
    return mask_high, th

def compute_mean_HOG_direction(region, mask_high):
    """Tính mean HOG direction từ high-entropy regions"""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Tính HOG
    hog_features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8,8),
                                  cells_per_block=(2,2), visualize=True, feature_vector=False)
    # Resize mask
    mh, mw = hog_image.shape
    mask_resized = cv2.resize(mask_high, (mw, mh))
    
    # Tính gradient
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    
    # Masked
    mag_masked = mag * mask_high
    angle_masked = angle * mask_high
    
    # Mean direction
    mean_dx = np.mean(np.cos(angle_masked[mask_high > 0]))
    mean_dy = np.mean(np.sin(angle_masked[mask_high > 0]))
    return mean_dx, mean_dy, hog_image

def expand_bbox(bbox, direction, expand_ratio=0.15):
    """Expand bbox theo HOG direction"""
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    dx, dy = direction

    expand_x = int(dx * w * expand_ratio)
    expand_y = int(dy * h * expand_ratio)

    x1_new = max(0, x1 - expand_x)
    y1_new = max(0, y1 - expand_y)
    x2_new = x2 + expand_x
    y2_new = y2 + expand_y

    return [x1_new, y1_new, x2_new, y2_new]

def cluster_bboxes_with_mean_shift(refined_bboxes, image_shape, bandwidth=None):
    """
    Gom cụm refined bboxes bằng Mean Shift để tạo bbox tổng quát.
    Dựa trên center coordinates + size.
    """
    if not refined_bboxes:
        return []
    
    # Extract features: [cx, cy, w, h]
    features = []
    for bbox in refined_bboxes:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        features.append([cx, cy, w, h])
    
    features = np.array(features)
    
    # Estimate bandwidth nếu không cung cấp
    if bandwidth is None:
        bandwidth = estimate_bandwidth(features, quantile=0.2)
    
    # Mean Shift clustering
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(features)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    # Tạo bbox tổng quát cho mỗi cluster
    clustered_bboxes = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = features[labels == label]
        if len(cluster_points) == 0:
            continue
        
        # Mean center và size
        mean_cx, mean_cy, mean_w, mean_h = cluster_centers[label]
        
        # Expand để cover all points in cluster
        min_cx = cluster_points[:, 0].min()
        max_cx = cluster_points[:, 0].max()
        min_cy = cluster_points[:, 1].min()
        max_cy = cluster_points[:, 1].max()
        
        # Bbox tổng quát
        x1 = int(min_cx - mean_w / 2)
        y1 = int(min_cy - mean_h / 2)
        x2 = int(max_cx + mean_w / 2)
        y2 = int(max_cy + mean_h / 2)
        
        # Clip to image
        h_img, w_img = image_shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)
        
        clustered_bboxes.append([x1, y1, x2, y2])
    
    return clustered_bboxes

def process_image(image_path, detector, refiner, output_dir):
    """Pipeline đầy đủ: Detect → Refine → Cluster"""
    print(f"Processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading {image_path}")
        return
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Step 1: Detection (sử dụng SAM2/Grounding DINO)
    text_prompt = "car"  # Hoặc prompt phù hợp
    detections = detector.predict(image, text_prompt, box_threshold=0.05, text_threshold=0.05)
    
    # Extract bboxes (giả sử detections có 'boxes')
    initial_bboxes = []
    if hasattr(detections, 'boxes'):
        for box in detections.boxes:
            x1, y1, x2, y2 = box.tolist()
            initial_bboxes.append([x1, y1, x2, y2])
    
    if not initial_bboxes:
        print("No detections found")
        return
    
    # Step 2: Refine từng bbox bằng entropy + HOG
    refined_bboxes = []
    for bbox in initial_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
        
        # Tính entropy
        entropy_map = compute_entropy_map(crop, window_size=20)
        mask_high, th = threshold_entropy(entropy_map, method="mean")
        
        # Tính HOG direction
        mean_dx, mean_dy, hog_img = compute_mean_HOG_direction(crop, mask_high)
        
        # Expand bbox
        new_bbox = expand_bbox(bbox, (mean_dx, mean_dy), expand_ratio=0.25)
        refined_bboxes.append(new_bbox)
    
    # Step 3: Cluster refined bboxes bằng Mean Shift
    clustered_bboxes = cluster_bboxes_with_mean_shift(refined_bboxes, image.shape)
    
    # Step 4: Visualize và save
    img_viz = image.copy()
    
    # Vẽ initial bboxes (xanh)
    for bbox in initial_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
    
    # Vẽ refined bboxes (đỏ)
    for bbox in refined_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
    
    # Vẽ clustered bboxes (xanh lá)
    for bbox in clustered_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"{base_name}_pipeline.png")
    cv2.imwrite(viz_path, img_viz)
    print(f"Saved visualization: {viz_path}")
    
    # Save results as JSON
    import json
    results = {
        'initial_bboxes': initial_bboxes,
        'refined_bboxes': refined_bboxes,
        'clustered_bboxes': clustered_bboxes
    }
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {json_path}")

# Main pipeline
if __name__ == "__main__":
    # Init detector và refiner
    detector = LoadDetector()
    refiner = BBoxRefinement()
    
    # Folder input/output
    input_folder = '/home/serverai/ltdoanh/LayoutGeneration/data/samples/keyframe4check/42853_keyframes'
    output_dir = 'pipeline_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process từng image
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, detector, refiner, output_dir)
    
    print("Pipeline complete!")