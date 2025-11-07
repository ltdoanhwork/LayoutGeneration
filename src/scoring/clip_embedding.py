import torch
import clip
from PIL import Image


def get_clip_embedding(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features.cpu().numpy().squeeze()  # (dim_clip,)