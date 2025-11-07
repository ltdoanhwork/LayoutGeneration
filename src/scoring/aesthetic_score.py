import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO
from utils.process_images import load_image_from_path

NAME_TO_HANDLE = {
    # Model trained on SPAQ dataset: https://github.com/h4nwei/SPAQ
    'spaq': '/kaggle/input/musiq/tensorflow2/spaq/1',

    # Model trained on KonIQ-10K dataset: http://database.mmsp-kn.de/koniq-10k-database.html
    'koniq': '/kaggle/input/musiq/tensorflow2/koniq-10k/1',

    # Model trained on PaQ2PiQ dataset: https://github.com/baidut/PaQ-2-PiQ
    'paq2piq': '/kaggle/input/musiq/tensorflow2/paq2piq/1',

    # Model trained on AVA dataset: https://ieeexplore.ieee.org/document/6247954
    'ava': './repos/ava_v1',
}

def _ensure_rgb_bytes(b: bytes) -> bytes:
    im = Image.open(BytesIO(b))
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def get_aesthetic_score(image_url, model_name='ava'):
    try:
        model_handle = NAME_TO_HANDLE[model_name]
    except KeyError:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(NAME_TO_HANDLE.keys())}")
    
    model = hub.load(model_handle)
    predict_fn = model.signatures['serving_default']

    image, image_bytes = load_image_from_path(image_url)
    if isinstance(image_bytes, str):
        image_bytes = image_bytes.encode('utf-8')
    input_name = list(predict_fn.structured_input_signature[1].keys())[0]

    # try original bytes first, fallback to re-encoded RGB bytes on failure
    tensor_bytes = tf.constant(image_bytes, dtype=tf.string)
    try:
        prediction = predict_fn(**{input_name: tensor_bytes})
    except Exception as e:
        print("First attempt failed, re-encoding to RGB and retrying:", e)
        safe_bytes = _ensure_rgb_bytes(image_bytes)
        tensor_bytes = tf.constant(safe_bytes, dtype=tf.string)
        prediction = predict_fn(**{input_name: tensor_bytes})

    out_key = list(prediction.keys())[0]
    score = float(prediction[out_key].numpy().item())
    return score


if __name__ == "__main__":
    test_image_url = "/home/serverai/ltdoanh/LayoutGeneration/repos/Colla/output_dir/baby/collage.png"
    score = get_aesthetic_score(test_image_url, model_name='ava')
    print(f"Aesthetic score for test image: {score:.4f}")