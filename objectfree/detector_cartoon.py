from ultralytics import YOLOE
import yaml
import os
import torch

class DetectorCartoon:
    def __init__(self,if __name__ == "__main__":
    # Example usage with config file
    config_path = "/home/serverai/ltdoanh/LayoutGeneration/objectfree/yoloe/detector_config.yaml"
    detector = DetectorCartoon(config_path=config_path)
    
    # Run detection based on config (có thể save nếu muốn)
    results = detector.forward(save_results=True)
    print(f"Detection completed. Found {len(results[0].boxes)} objects")
    results.show()path: str):
        """
        Initialize detector with configuration from YAML file.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        
        # Extract configuration values
        self.model_path = self.config.get('model_path')
        self.input_path = self.config.get('input_path')  # For single image, batch folder, or video
        self.threshold = self.config.get('threshold', 0.25)
        self.prompt = self.config.get('prompt', 'characters in cartoon')
        self.save_path = self.config.get('save_path')
        self.type_content = self.config.get('type_content', 'image')  # 'image' or 'video'
        
        # Validate paths
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path not found: {self.input_path}")
        
        # Load YOLOE model
        self.model = YOLOE(self.model_path)
        
        # Load saved prompt embeddings instead of generating new ones
        pe_path = self.config.get('pe_path', 'character-pe.pt')
        if os.path.exists(pe_path):
            pe_data = torch.load(pe_path)
            saved_names = pe_data['names']
            saved_pe = pe_data['pe']
            # Use saved embeddings for the prompt
            self.model.set_classes(saved_names, saved_pe)
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
                return self.model.predict(
                    source=self.input_path, 
                    conf=self.threshold,
                    save=save_dir is not None,
                    save_dir=save_dir
                )
    
    def _predict_batch(self, image_paths: list = None, folder_path: str = None, save_dir: str = None):
        """
        Internal method for batch prediction.
        """
        if folder_path:
            # Get all image files from folder
            image_paths = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
        
        if not image_paths:
            raise ValueError("Either image_paths or folder_path must be provided")
        
        results = self.model.predict(
            source=image_paths,
            conf=self.threshold,
            save=save_dir is not None,
            save_dir=save_dir
        )
        return results
    
    def _predict_video(self, save_dir: str = None):
        """
        Internal method for video prediction.
        """
        results = self.model.predict(
            source=self.input_path,
            conf=self.threshold,
            save=save_dir is not None,
            save_dir=save_dir
        )
        return results

if __name__ == "__main__":
    # Example usage with config file
    config_path = "/home/serverai/ltdoanh/LayoutGeneration/objectfree/detector_config.yaml"
    detector = DetectorCartoon(config_path=config_path)
    
    # Run detection based on config (có thể save nếu muốn)
    results = detector.forward(save_results=True)
    results.show()