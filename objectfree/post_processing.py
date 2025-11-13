import torch
import numpy as np
import supervision as sv
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn.functional as F
import torchvision.ops as ops

class PostProcessor:

    def __init__(self, device: str = "cuda", boxes: list = None, scores: list = None):
        self.device = device
        self.boxes = boxes
        self.scores = scores

        # Initialize image embedding model for similarity filtering
        self.embedding_model = resnet50(pretrained=True)
        self.embedding_model = torch.nn.Sequential(*list(self.embedding_model.children())[:-1])  # Remove classification layer
        self.embedding_model.to(device)
        self.embedding_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _apply_nms(self, boxes, scores, image, iou_threshold=0.5, similarity_threshold=0.4):
        """
        Apply image similarity filtering followed by Non-Maximum Suppression
        First filter boxes where cropped image similarity to full image < threshold,
        then apply standard NMS
        Args:
            boxes: numpy array (N, 4) in xyxy format
            scores: numpy array (N,)
            image: PIL Image object
            iou_threshold: IoU threshold for NMS
            similarity_threshold: cosine similarity threshold for filtering
        Returns:
            keep_indices: list of indices to keep
        """
        if len(boxes) == 0:
            return []

        # Get full image embedding
        full_image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            full_embedding = self.embedding_model(full_image_tensor).squeeze()

        # Filter by image similarity
        valid_indices = []
        for i, box in enumerate(boxes):
            # Crop image using bounding box
            x1, y1, x2, y2 = map(int, box)
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)

            if x2 <= x1 or y2 <= y1:
                continue  # Invalid box

            cropped_image = image.crop((x1, y1, x2, y2))

            # Get cropped image embedding
            cropped_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cropped_embedding = self.embedding_model(cropped_tensor).squeeze()

            # Compute cosine similarity
            similarity = F.cosine_similarity(full_embedding, cropped_embedding, dim=0).item()

            if similarity >= similarity_threshold:
                valid_indices.append(i)

        if not valid_indices:
            return []

        # Filter boxes and scores
        filtered_boxes = boxes[valid_indices]
        filtered_scores = scores[valid_indices]

        # Apply standard NMS
        boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32).to(self.device)
        scores_tensor = torch.tensor(filtered_scores, dtype=torch.float32).to(self.device)

        keep_indices = ops.nms(
            boxes=boxes_tensor,
            scores=scores_tensor,
            iou_threshold=iou_threshold,
        )

        # Map back to original indices
        original_keep_indices = np.array(valid_indices)[keep_indices.cpu().numpy()]
        return original_keep_indices.tolist()

    def _apply_wbf(self, boxes, scores, iou_threshold=0.55):
        """
        Apply Weighted Box Fusion to combine overlapping detections
        Args:
            boxes: numpy array (N, 4) in xyxy format
            scores: numpy array (N,)
            iou_threshold: IoU threshold for matching boxes
        Returns:
            fused_boxes: numpy array (M, 4) - fused boxes
            fused_scores: numpy array (M,) - fused scores
        """
        if len(boxes) == 0:
            return boxes, scores
        
        
        # Group boxes by similarity (clustering)
        clustered_boxes = []
        clustered_scores = []
        used = set()
        
        for i in range(len(boxes)):
            if i in used:
                continue
            
            cluster_boxes = [boxes[i]]
            cluster_scores = [scores[i]]
            used.add(i)
            
            # Find similar boxes
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                
                iou = self._compute_iou(boxes[i], boxes[j])
                if iou > iou_threshold:
                    cluster_boxes.append(boxes[j])
                    cluster_scores.append(scores[j])
                    used.add(j)
            
            clustered_boxes.append(cluster_boxes)
            clustered_scores.append(cluster_scores)
        
        # Compute weighted boxes
        fused_boxes = []
        fused_scores = []
        
        for cluster_box, cluster_score in zip(clustered_boxes, clustered_scores):
            cluster_box = np.array(cluster_box)
            cluster_score = np.array(cluster_score)
            
            # Weighted average
            weight_sum = cluster_score.sum()
            weighted_box = (cluster_box * cluster_score[:, np.newaxis]).sum(axis=0) / weight_sum
            avg_score = cluster_score.mean()
            
            fused_boxes.append(weighted_box)
            fused_scores.append(avg_score)
        
        fused_boxes = np.array(fused_boxes)
        fused_scores = np.array(fused_scores)
        
        return fused_boxes, fused_scores

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes in xyxy format"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def _filter_similarity(self, sim_scores, threshold=0.3):
        """
        Filter detections based on similarity score threshold
        Xóa crops khi similarity với toàn cục < threshold
        
        Args:
            sim_scores: numpy array hoặc list of similarity scores
            threshold: minimum similarity to keep (default 0.3)
        Returns:
            keep_indices: list of indices to keep
        """
        sim_scores = np.array(sim_scores)
        keep_indices = np.where(sim_scores >= threshold)[0]
        return keep_indices.tolist()

    def filter_detections_by_similarity(self, boxes, scores, sim_scores, sim_threshold=0.3):
        """
        Filter boxes and scores by similarity threshold
        Giữ lại detection nếu similarity cao, xóa nếu không giống nội dung toàn cục
        
        Args:
            boxes: numpy array (N, 4) in xyxy format
            scores: numpy array (N,) - confidence scores    
            sim_scores: numpy array (N,) - similarity scores (toàn cục vs crop)
            sim_threshold: minimum similarity to keep (default 0.3)
        Returns:
            filtered_boxes: filtered boxes
            filtered_scores: filtered scores
            filtered_sim_scores: filtered similarity scores
        """
        if len(boxes) == 0:
            return boxes, scores, sim_scores
        
        # Filter by similarity threshold
        keep_indices = self._filter_similarity(sim_scores, sim_threshold)
        
        if len(keep_indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        filtered_boxes = boxes[keep_indices]
        filtered_scores = scores[keep_indices]
        filtered_sim_scores = sim_scores[keep_indices]
        
        return filtered_boxes, filtered_scores, filtered_sim_scores                     