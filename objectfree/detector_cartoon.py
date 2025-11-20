from ultralytics import YOLOE
import yaml
import os
import torch
from tqdm import tqdm

class DetectorCartoon:
    def __init__(self, config_path: str):
        """
        Initialize detector with configuration from YAML file.

        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_path)

        # Get config directory for resolving relative paths
        self.config_dir = os.path.dirname(os.path.abspath(config_path))

        # Extract configuration values and resolve relative paths
        self.model_path = self._resolve_path(self.config.get('model_path'))
        self.input_path = self._resolve_path(self.config.get('input_path'))  # For single image, batch folder, or video
        self.threshold = self.config.get('threshold', 0.25)
        self.prompt = self.config.get('prompt', 'characters in cartoon')
        self.save_path = self._resolve_path(self.config.get('save_path'))
        self.type_content = self.config.get('type_content', 'image')  # 'image' or 'video'
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Validate paths
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path not found: {self.input_path}")

        # Load YOLOE model with specified device
        print(f"Loading model on device: {self.device}")
        self.model = YOLOE(self.model_path)

        # Load saved prompt embeddings instead of generating new ones
        pe_path = self.config.get('pe_path', 'character-pe.pt')
        if os.path.exists(pe_path):
            pe_data = torch.load(pe_path, map_location=self.device)
            saved_names = pe_data['names']
            saved_pe = pe_data['pe'].to(self.device)
            # For single-class model, use the first embedding or average them
            # Use the first class embedding for single-class detection
            single_pe = saved_pe[:, 0:1, :]  # Take first class embedding
            self.model.set_classes(['character'], single_pe)  # Single class
            print(f"Loaded {len(saved_names)} character classes, using first one for detection")
        else:
            # Fallback: generate embeddings (but this requires mobileclip model)
            tpe = self.model.get_text_pe([self.prompt])
            self.model.set_classes([self.prompt], tpe)

    @staticmethod
    def _load_config(yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _resolve_path(self, path: str):
        """
        Resolve relative paths relative to config directory.
        Absolute paths are returned as-is.

        Args:
            path (str): Path to resolve

        Returns:
            str: Resolved absolute path
        """
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        # Resolve relative to config directory
        return os.path.abspath(os.path.join(self.config_dir, path))

    def forward(self, save_results: bool = False):
        """
        Run detection based on type_content.
        - If 'video': predict on video
        - If 'image': predict on batch images (if folder) or single image

        Args:
            save_results (bool): Whether to save results
        """
        save_dir = self.save_path if save_results and self.save_path else None

        if self.type_content == 'video':
            return self._predict_video(save_dir)
        else:  # image
            if os.path.isdir(self.input_path):
                return self._predict_batch(folder_path=self.input_path, save_dir=save_dir)
            else:
                # Single image prediction
                results = self.model.predict(
                    source=self.input_path,
                    conf=self.threshold,
                    save=save_dir is not None,
                    save_dir=save_dir,
                    stream=False,  # Single image, no need for stream
                    device=self.device
                )
                return results

    def _predict_batch(self, image_paths: list = None, folder_path: str = None, save_dir: str = None, max_images: int = 10):
        """
        Internal method for batch prediction.

        Args:
            max_images (int): Maximum number of images to process for testing
        """
        if folder_path:
            # Get all image files from folder
            all_image_paths = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            # Limit number of images for testing
            image_paths = all_image_paths[:max_images]
            print(f"Processing {len(image_paths)} out of {len(all_image_paths)} images")

        if not image_paths:
            raise ValueError("Either image_paths or folder_path must be provided")

        # Run prediction with progress bar
        results_generator = self.model.predict(
            source=image_paths,
            conf=self.threshold,
            save=save_dir is not None,
            save_dir=save_dir,
            stream=True,  # Keep stream for batch processing
            device=self.device
        )

        # Process results with progress bar
        results_list = []
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            for result in results_generator:
                results_list.append(result)
                pbar.update(1)

        return results_list

    def _predict_video(self, save_dir: str = None):
        """
        Internal method for video prediction.
        """
        results_generator = self.model.predict(
            source=self.input_path,
            conf=self.threshold,
            save=save_dir is not None,
            save_dir=save_dir,
            stream=True,
            device=self.device
        )

        # For video, we can either return the generator or collect results
        # For now, return the generator (caller can iterate if needed)
        return results_generator

if __name__ == "__main__":
    # Example usage with config file
    config_path = "yoloe/detector_config.yaml"
    detector = DetectorCartoon(config_path=config_path)

    # Run detection based on config (có thể save nếu muốn)
    results = detector.forward(save_results=True)

    # Handle different result types
    if detector.type_content == 'video':
        print("Video prediction completed. Results saved to output directory.")
    else:
        if isinstance(results, list):
            # Batch results
            print(f"Processed {len(results)} images")
            total_detections = sum(len(r.boxes) for r in results if hasattr(r, 'boxes'))
            print(f"Total detections: {total_detections}")
            print(f"Results saved to: {detector.save_path}")
        else:
            # Single image result
            print("Prediction completed. Check output directory for results.")
            print(f"Results saved to: {detector.save_path}")
