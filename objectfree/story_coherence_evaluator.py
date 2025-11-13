"""
Story Coherence Evaluator using BLIP
Đánh giá độ phù hợp về mặt câu chuyện giữa ảnh gốc và các vùng crop
"""

import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import json
import os
from typing import List, Dict, Tuple, Optional
import logging
import glob


class StoryCoherenceEvaluator:
    """
    Đánh giá độ liên quan về câu chuyện giữa ảnh gốc và các vùng crop
    
    Pipeline:
    1. BLIP caption cho ảnh gốc → hiểu bối cảnh tổng thể
    2. BLIP caption cho từng vùng crop (từ object detection)
    3. So sánh semantic similarity giữa captions
    4. Tính điểm coherence cho mỗi crop
    """
    
    def __init__(
        self, 
        blip_model_name: str = "Salesforce/blip-image-captioning-large",
        sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        output_Blip: str = None
    ):
        """
        Args:
            blip_model_name: Tên model BLIP trên HuggingFace
            sentence_model_name: Tên model để tính similarity
            device: 'cuda' hoặc 'cpu', mặc định tự động
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_Blip = output_Blip
        # Load BLIP cho captioning
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_name
        ).to(self.device).eval()


        # Load Sentence Transformer cho semantic similarity
        self.sentence_model = SentenceTransformer(sentence_model_name, device=self.device)
    
    def generate_caption(
        self, 
        image: Image.Image, 
        max_new_tokens: int = 50
    ) -> str:
        """
        Sinh caption cho một ảnh bằng BLIP
        
        Args:
            image: PIL Image
            max_new_tokens: Độ dài tối đa của caption
            
        Returns:
            caption: Mô tả ảnh bằng tiếng Anh
        """
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def crop_image_from_bbox(
        self, 
        image: Image.Image, 
        bbox: List[float]
    ) -> Image.Image:
        """
        Crop ảnh theo bounding box
        
        Args:
            image: PIL Image gốc
            bbox: [x1, y1, x2, y2] - tọa độ pixel
            
        Returns:
            cropped_image: PIL Image đã crop
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        # Đảm bảo tọa độ hợp lệ
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        
        return image.crop((x1, y1, x2, y2))
    
    def compute_similarity(
        self, 
        caption1: str, 
        caption2: str
    ) -> float:
        """
        Tính độ tương đồng semantic giữa 2 captions
        
        Args:
            caption1: Caption ảnh gốc (toàn cục)
            caption2: Caption crop
            
        Returns:
            similarity: Điểm từ 0 đến 1 (1 = giống nhau hoàn toàn)
        """
        # Encode captions thành embeddings
        emb1 = self.sentence_model.encode(caption1, convert_to_tensor=True)
        emb2 = self.sentence_model.encode(caption2, convert_to_tensor=True)
        
        # Tính cosine similarity
        similarity = util.cos_sim(emb1, emb2).item()
        return similarity
    
    def filter_by_similarity(self, similarity: float, threshold: float = 0.3) -> bool:
        """
        Filter crop: giữ lại nếu similarity cao (crop liên quan đến toàn cục)
        Xóa nếu similarity thấp (crop không giống nội dung toàn cục)
        
        Args:
            similarity: Similarity score từ 0-1
            threshold: Ngưỡng tối thiểu để giữ crop (default 0.3)
            
        Returns:
            True nếu giữ, False nếu xóa
        """
        return similarity >= threshold
    
    def assign_bbox_color(self, det_idx: int) -> str:
        """
        Gán màu cho bbox dựa trên detection index
        
        Args:
            det_idx: Index của detection
            
        Returns:
            color: Màu hex string
        """
        colors = [
            '#FF0000',  # Red
            '#00FF00',  # Green  
            '#0000FF',  # Blue
            '#FFFF00',  # Yellow
            '#FF00FF',  # Magenta
            '#00FFFF',  # Cyan
            '#FFA500',  # Orange
            '#800080',  # Purple
            '#FFC0CB',  # Pink
            '#A52A2A',  # Brown
            '#808080',  # Gray
            '#000000',  # Black
        ]
        return colors[det_idx % len(colors)]
    
    
    def evaluate_batch(
        self,
        detection_results_json: str,
        keyframes_folder: str,
        output_dir: str,
        save_crops: bool = False,  # Changed default to False
        similarity_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Đánh giá batch từ file detection_results.json
        
        Args:
            detection_results_json: Đường dẫn file JSON từ object detection
            keyframes_folder: Đường dẫn folder chứa ảnh gốc keyframes
            output_dir: Thư mục lưu kết quả
            save_crops: Có lưu ảnh crop không (default False)
            similarity_threshold: Ngưỡng similarity để filter crops
            
        Returns:
            all_results: List of result dicts
        """
        # Load detection results
        with open(detection_results_json, 'r', encoding='utf-8') as f:
            det_data = json.load(f)
        
        results_list = det_data.get('results', [])
        
        # Lấy tất cả ảnh gốc trong keyframes folder
        file_images = ["*.jpg","*.png","*.jpeg"]
        keyframe_images = []
        for pattern in file_images:
            keyframe_images.extend(glob.glob(os.path.join(keyframes_folder, pattern)))
        keyframe_images = sorted(keyframe_images)        
        all_results = []

        for idx, res in enumerate(results_list):

            img_path = keyframe_images[idx]
            
            detections = res.get('detections', [])
            
            # Đánh giá
            result = self.evaluate_single_image(
                image_path=img_path,
                detections=detections,
                output_dir=output_dir,
                save_crops=save_crops,
                similarity_threshold=similarity_threshold
            )
            
            all_results.append(result)
        
        # Lưu summary
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "story_coherence_summary.json")
        summary = {
            'total_images': len(all_results),
            'avg_similarity_across_all': round(np.mean([r['avg_similarity'] for r in all_results]), 4) if all_results else 0.0,
            'results': all_results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return all_results
    
    def evaluate_single_image(
        self,
        image_path: str,
        detections: List[Dict],
        output_dir: str,
        save_crops: bool = False,  # Changed default to False
        similarity_threshold: float = 0.3
    ) -> Dict:
        """
        Đánh giá story coherence cho một ảnh
        
        Args:
            image_path: Đường dẫn ảnh
            detections: List of detection dicts
            output_dir: Thư mục lưu kết quả
            save_crops: Có lưu ảnh crop không (default False)
            similarity_threshold: Ngưỡng similarity để filter crops
            
        Returns:
            result: Dict chứa kết quả đánh giá
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Generate caption cho toàn bộ ảnh
        full_caption = self.generate_caption(image)
        
        crop_results = []
        similarities = []
        
        for det_idx, det in enumerate(detections):
            bbox = det['bbox']  # [x1, y1, x2, y2]
            
            # Crop image
            crop = self.crop_image_from_bbox(image, bbox)
            
            # Generate caption cho crop
            crop_caption = self.generate_caption(crop)
            
            # Compute similarity
            similarity = self.compute_similarity(full_caption, crop_caption)
            similarities.append(similarity)
            
            # Filter by similarity
            keep = self.filter_by_similarity(similarity, threshold=similarity_threshold)
            
            # Assign color for bbox
            bbox_color = self.assign_bbox_color(det_idx)
            
            crop_result = {
                'detection_id': det_idx,
                'bbox': bbox,
                'bbox_color': bbox_color,
                'full_caption': full_caption,
                'crop_caption': crop_caption,
                'similarity': round(similarity, 4),
                'keep': keep
            }
            
            # Chỉ lưu crop image nếu được yêu cầu VÀ được keep
            if save_crops and keep:
                # Lưu crop
                os.makedirs(output_dir, exist_ok=True)
                crop_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_crop_{det_idx}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                crop.save(crop_path)
                crop_result['crop_path'] = crop_path
            
            # Luôn thêm vào results, bất kể có keep hay không
            crop_results.append(crop_result)
        
        # Tính avg similarity
        avg_similarity = round(np.mean(similarities), 4) if similarities else 0.0
        
        result = {
            'image_path': image_path,
            'full_caption': full_caption,
            'num_detections': len(detections),
            'num_kept_crops': sum(1 for r in crop_results if r['keep']),
            'avg_similarity': avg_similarity,
            'crop_results': crop_results
        }
        
        return result


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Khởi tạo evaluator
    evaluator = StoryCoherenceEvaluator(
        blip_model_name="Salesforce/blip-image-captioning-large",
        device="cuda"
    )