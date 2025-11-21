# Cartoon Character Detection System

A comprehensive system for detecting cartoon characters in images and videos using YOLOE (YOLO with Open Vocabulary) and advanced prompt engineering.

## Overv### Output Locations

Results are saved to:
- **Detection Results**: `/home/serverai/ltdoanh/LayoutGeneration/outputs/objects_free/`
- **Annotated Images**: `outputs/objects_free/predict/` subdirectory
- **No automatic image display** - results are saved silently for pipeline integration provides:
- **Object Detection**: Detect cartoon characters using YOLOE with custom-trained models
- **Batch Processing**: Process multiple images or videos with progress monitoring
- **CUDA Support**: GPU acceleration for faster inference
- **Configurable Prompts**: Use text embeddings for character-specific detection
- **Flexible Input**: Support for single images, image folders, and video files
- **Organized Weights**: All model weights centralized in `weight_model/` directory

## Project Structure

```
objectfree/
├── detector_cartoon.py          # Main detector class
├── detector_config.yaml         # Configuration file
├── README.md                    # This documentation
├── weight_model/                # 📁 Centralized model weights
│   ├── character-pe.pt         # Prompt embeddings for character detection
│   ├── mobileclip_blt.pt       # MobileCLIP text model
│   └── yoloe/
│       ├── weights/            # YOLOE model weights
│       │   ├── best_general.pt # Best general model
│       │   ├── last.pt         # Latest training checkpoint
│       │   └── last_mosaic.pt  # Mosaic augmentation checkpoint
│       └── train9_weights/     # Alternative training weights
├── yoloe/                       # YOLOE training and inference code
├── annotations/                 # Dataset annotations
├── BLIP2/                       # BLIP-2 integration
├── Grounded-SAM-2/              # Grounded SAM integration
└── results/                     # Output directory
```

## Quick Start

### Prerequisites

```bash
pip install ultralytics torch torchvision tqdm pyyaml
```

### Weight Management

All model weights are organized in the `weight_model/` directory:
- **character-pe.pt**: Prompt embeddings for character detection
- **mobileclip_blt.pt**: MobileCLIP text model for prompt processing
- **yoloe/weights/**: YOLOE model checkpoints

### Basic Usage

1. **Configure Detection Settings**

Edit `detector_config.yaml`:

```yaml
model_path: "weight_model/yoloe/weights/best_general.pt"
input_path: "../data/images/"                           # Input images/videos
threshold: 0.25                                        # Detection confidence threshold
prompt: "characters in cartoon"                        # Detection prompt
save_path: "/home/serverai/ltdoanh/LayoutGeneration/outputs/objects_free"  # Output directory
type_content: "image"                                  # 'image' or 'video'
device: "cuda"                                         # 'cuda' or 'cpu'
pe_path: "weight_model/character-pe.pt"               # Prompt embeddings
```

2. **Run Detection**

```bash
cd objectfree
python detector_cartoon.py
```

## Configuration Options

### Model Configuration

- **model_path**: Path to YOLOE model weights (`.pt` file)
  - Default: `"weight_model/yoloe/weights/best_general.pt"`
- **pe_path**: Path to saved prompt embeddings (`.pt` file)
  - Default: `"weight_model/character-pe.pt"`
- **device**: Computing device (`'cuda'` for GPU, `'cpu'` for CPU)
  - Auto-detected: `'cuda'` if available, otherwise `'cpu'`

### Input/Output Configuration

- **input_path**: Path to input file/folder
  - Single image: `"path/to/image.jpg"`
  - Image folder: `"path/to/images/"`
  - Video file: `"path/to/video.mp4"`
- **save_path**: Directory to save detection results
- **type_content**: Input type (`'image'` or `'video'`)

### Detection Parameters

- **threshold**: Confidence threshold (0.0-1.0)
- **prompt**: Text prompt for character detection

## Advanced Usage

### Using the Detector Class

```python
from detector_cartoon import DetectorCartoon

# Initialize detector
detector = DetectorCartoon(config_path="detector_config.yaml")

# Run detection
results = detector.forward(save_results=True)

# Process results
if isinstance(results, list):
    print(f"Detected {len(results)} objects")
    for result in results:
        print(f"Boxes: {len(result.boxes)}")
```

### Batch Processing

The system automatically handles different input types:

- **Single Image**: Returns single result object
- **Image Folder**: Returns list of results with progress bar
- **Video**: Returns result generator for frame-by-frame processing

### Custom Prompts

The system uses pre-computed prompt embeddings for efficient detection. To use custom prompts:

1. Generate new embeddings using the training script
2. Update `pe_path` in config
3. Modify `prompt` parameter

## Weight Management

### Directory Structure

```
weight_model/
├── character-pe.pt              # Character detection prompt embeddings
├── mobileclip_blt.pt            # MobileCLIP text model
└── yoloe/
    ├── weights/                 # Primary model weights
    │   ├── best_general.pt      # Best performing model
    │   ├── last.pt              # Latest training checkpoint
    │   └── last_mosaic.pt       # Checkpoint with mosaic augmentation
    └── train9_weights/          # Alternative training run
        ├── best.pt
        ├── last.pt
        └── last_mosaic.pt
```

### Model Versions

- **best_general.pt**: Recommended for general cartoon character detection
- **train9_weights/best.pt**: Alternative model trained on different dataset
- **character-pe.pt**: Contains embeddings for 82 character classes

### Updating Weights

To use different model weights:

1. Copy new weights to `weight_model/yoloe/weights/`
2. Update `model_path` in config file
3. Restart detector

### Backup Strategy

- Keep multiple training runs in separate folders
- Use descriptive names for different model versions
- Regularly backup `weight_model/` directory

## Training

To train a custom YOLOE model:

```bash
cd yoloe
python train_cartoon.py
```

Training requires:
- Annotated dataset in YOLO format
- Character class definitions
- Sufficient GPU memory (recommended: 8GB+)

### After Training

1. Copy trained weights to `weight_model/yoloe/weights/`
2. Update prompt embeddings if new classes were added
3. Test with validation data
4. Update config files to use new weights

## Output Formats

### Detection Results

Results include:
- **Bounding boxes**: Character locations
- **Confidence scores**: Detection confidence
- **Class labels**: Character categories
- **Visualizations**: Annotated images (when `save_results=True`)

### Saved Files

- `results/` directory contains:
  - Annotated images with bounding boxes
  - Detection statistics
  - Video processing results

## Performance Optimization

### GPU Acceleration

- Set `device: "cuda"` for GPU processing
- Ensure CUDA-compatible GPU is available
- Monitor GPU memory usage during batch processing

### Batch Processing Tips

- Use `max_images` parameter to limit processing
- Progress bars show processing status
- Results are streamed for memory efficiency

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify model path exists
   - Check model compatibility with YOLOE version

2. **CUDA Errors**
   - Ensure CUDA drivers are installed
   - Check GPU memory availability
   - Fallback to CPU: `device: "cpu"`

3. **Path Errors**
   - Use relative paths from project root
   - Ensure input files exist
   - Check write permissions for output directory

4. **Memory Issues**
   - Reduce batch size for large images
   - Use CPU for memory-constrained systems
   - Process videos frame-by-frame

### Debug Mode

Enable verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Dependencies

- **ultralytics**: YOLOE implementation
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities
- **tqdm**: Progress bar visualization
- **pyyaml**: Configuration file parsing
- **CUDA**: GPU acceleration (optional)

## License

This project is part of the LayoutGeneration system for cartoon analysis and processing.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify configuration files
3. Test with sample data
4. Review error messages carefully