
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import cv2

def load_image_from_path(img_path):
  """Returns an image and the raw image bytes (image, image_bytes)."""
  # Read raw bytes from local file path
  with open(img_path, 'rb') as f:
    image_bytes = f.read()
  # Use BytesIO so Pillow can open the bytes consistently
  image = Image.open(BytesIO(image_bytes))
  return image, image_bytes


def show_image(image, title=''):
  image_size = image.size
  plt.imshow(image)
  plt.axis('on')
  plt.title(title)
  plt.show()


def _to_numpy_rgb(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Ensure (H,W,3) uint8 RGB numpy array."""
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        return np.array(img)
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            # gray -> RGB
            return np.repeat(img[..., None], 3, axis=2)
        if img.shape[2] == 4:
            # RGBA -> RGB
            return img[:, :, :3]
        return img
    else:
        raise TypeError("Unsupported image type. Use PIL.Image or np.ndarray.")


def _resize_keep_ar(img: np.ndarray, longer: int = 512) -> np.ndarray:
    """Resize keeping aspect ratio so that max(H,W)=longer."""
    h, w = img.shape[:2]
    s = longer / max(h, w)
    if s <= 1.0:
        nh, nw = int(round(h * s)), int(round(w * s))
        if cv2 is not None:
            return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            return np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    return img