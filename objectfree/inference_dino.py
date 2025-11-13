import sys
import glob
import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import yaml
import torch.nn as nn
import json
import logging

# Add parent directory to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# Import PostProcessor for filtering methods
from post_processing import PostProcessor


class LoadDetector(nn.Module):
    def __init__(self, config_path, checkpoint_path, image_path, device, batch_size=1, output_dir="outputs"):
        super(LoadDetector, self).__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.image_path = image_path
        self.device = device
        self.batch_size = batch_size
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        self.predictor = None
        # Initialize PostProcessor for filtering methods
        self.post_processor = PostProcessor(device=self.device)

    def load_batch_image(self):
        """Load images from paths"""
        paths = list(self.image_path)

        images = []
        for p in paths[: self.batch_size]:
            images.append(Image.open(p).convert("RGB"))
        return images
    
    def Read_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.text_prompt = config['config']['TEXT_PROMPT']
        self.sam2_checkpoint = config['config']['SAM2_CHECKPOINT']
        self.sam2_model_config = config['config']['SAM2_MODEL_CONFIG']
        self.grounding_dino_model = config['config']['GROUNDING_DINO_MODEL']
        self.box_threshold = config['config']['BOX_THRESHOLD']
        self.text_threshold = config['config']['TEXT_THRESHOLD']
        self.nms_threshold = config['config'].get('NMS_THRESHOLD', 0.5)
        self.use_wbf = config['config'].get('USE_WBF', False)
        return self.text_prompt, self.sam2_checkpoint, self.sam2_model_config, self.grounding_dino_model, self.box_threshold, self.text_threshold
    
    def load_grounding_dino(self):
        """Load SAM2 and Grounding DINO models"""
        logger.info(f"Loading SAM2 from: {self.sam2_checkpoint}")
        sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 loaded")
        logger.info(f"Loading Grounding DINO: {self.grounding_dino_model}")
        processor = AutoProcessor.from_pretrained(self.grounding_dino_model)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model).to(self.device)
        logger.info("Grounding DINO loaded")
        return processor, grounding_model
 
    def _bboxes_to_list(self, boxes_xyxy, scores):
        """Convert numpy arrays to list of dicts"""
        results_bbox = []
        for (x1, y1, x2, y2), s in zip(boxes_xyxy.tolist(), scores.tolist()):
            results_bbox.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "score": float(s)})
        return results_bbox

    def forward(self):
        # Load config
        self.text_prompt, self.sam2_checkpoint, self.sam2_model_config, self.grounding_dino_model, self.box_threshold, self.text_threshold = self.Read_config()
        logger.info("Config loaded")

        # Load models
        processor, grounding_model = self.load_grounding_dino()

        # Load images
        images = self.load_batch_image()
        logger.info(f"Loaded {len(images)} images for detection")

        all_results = []

        for i, image in enumerate(images):
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            img_cv = image_bgr.copy()

            # Grounding DINO inference
            inputs = processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            # Post-process
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                target_sizes=[image.size[::-1]]
            )
            
            boxes_filt = results[0]["boxes"].cpu().numpy()
            labels = results[0]["labels"]
            logits_filt = results[0]["scores"].cpu().numpy()

            if len(boxes_filt) == 0:
                logger.info(f"No objects detected in image {i}")
                all_results.append({
                    "image_index": i,
                    "image_size": {"width": image.width, "height": image.height},
                    "detections": [],
                    "filter_method": "none"
                })
                continue

            # Apply filtering: NMS or WBF
            filter_method = "none"
            if self.use_wbf:
                logger.debug(f"Applying WBF (iou_threshold={self.nms_threshold})...")
                boxes_filt, logits_filt = self.post_processor._apply_wbf(boxes_filt, logits_filt, self.nms_threshold)
                filter_method = "wbf"
            else:
                logger.debug(f"Applying NMS (iou_threshold={self.nms_threshold})...")
                keep_indices = self.post_processor._apply_nms(boxes_filt, logits_filt, image, self.nms_threshold)
                boxes_filt = boxes_filt[keep_indices]
                logits_filt = logits_filt[keep_indices]
                filter_method = "nms"
            
            logger.info(f"Image {i}: Detected {len(boxes_filt)} objects after {filter_method.upper()}")

            boxes_xyxy = boxes_filt

            # Create detections
            detections = sv.Detections(
                xyxy=boxes_xyxy,
                confidence=logits_filt,
                class_id=np.zeros(len(boxes_xyxy), dtype=int),
            )

            # Annotate
            box_annotator = sv.BoxAnnotator(thickness=2)
            annotated_frame = box_annotator.annotate(scene=img_cv.copy(), detections=detections)

            label_list = [f"{label} {s:.2f}" for label, s in zip(labels[:len(logits_filt)], logits_filt)]
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
            annotated_image = label_annotator.annotate(annotated_frame, detections=detections, labels=label_list)

            # Save
            base_name = f"image_{i}"
            annotated_path = os.path.join(self.output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_image)
            boxes_only = box_annotator.annotate(scene=img_cv.copy(), detections=detections)
            boxes_path = os.path.join(self.output_dir, f"{base_name}_boxes.jpg")
            cv2.imwrite(boxes_path, boxes_only)
            logger.debug(f"Saved: {annotated_path}, {boxes_path}")

            det_list = self._bboxes_to_list(boxes_xyxy, logits_filt)
            all_results.append({
                "image_index": i,
                "image_size": {"width": image.width, "height": image.height},
                "annotated_image": os.path.abspath(annotated_path),
                "boxes_image": os.path.abspath(boxes_path),
                "detections": det_list,
                "filter_method": filter_method,
                "num_detections": len(det_list)
            })

        # Save JSON
        json_path = os.path.join(self.output_dir, "detection_results.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({"results": all_results, "text_prompt": self.text_prompt}, jf, ensure_ascii=False, indent=2)

        logger.info(f"Detection results saved to: {json_path}")
        return all_results
    
    def run_inference(self, keyframes_folder, output_dir):
        """Run inference on all keyframes in folder"""
        image_paths = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
        image_paths = [p for p in image_paths if "preview" not in os.path.basename(p).lower()]
        
        if len(image_paths) == 0:
            return
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.image_path = image_paths
        self.output_dir = output_dir
        self.batch_size = len(image_paths)
        
        logger.info("Starting inference...")
        results = self.forward()
        
        logger.info(f"Detection completed: {len(results)} images processed")
        return results