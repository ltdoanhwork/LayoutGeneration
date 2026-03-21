"""
AnimeSegmentation - inference-only wrapper around ISNet / U2Net / MODNet.
Stripped from https://github.com/SkyTNT/anime-segmentation (train.py).
"""

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .model import ISNetDIS, ISNetGTEncoder, U2NET, U2NET_full2, U2NET_lite2, MODNet, \
    InSPyReNet, InSPyReNet_Res2Net50, InSPyReNet_SwinB

net_names = [
    "isnet_is", "isnet", "isnet_gt",
    "u2net", "u2netl",
    "modnet",
    "inspyrnet_res", "inspyrnet_swin",
]


def get_net(net_name, img_size):
    if net_name in ("isnet", "isnet_is"):
        return ISNetDIS()
    elif net_name == "isnet_gt":
        return ISNetGTEncoder()
    elif net_name == "u2net":
        return U2NET_full2()
    elif net_name == "u2netl":
        return U2NET_lite2()
    elif net_name == "modnet":
        return MODNet()
    elif net_name == "inspyrnet_res":
        return InSPyReNet_Res2Net50(base_size=img_size)
    elif net_name == "inspyrnet_swin":
        return InSPyReNet_SwinB(base_size=img_size)
    raise NotImplementedError(f"Unknown net: {net_name}")


class AnimeSegmentation(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="anime_segmentation",
    repo_url="https://github.com/SkyTNT/anime-segmentation",
    tags=["image-segmentation"],
):
    def __init__(self, net_name: str, img_size=None):
        super().__init__()
        assert net_name in net_names, f"net_name must be one of {net_names}"
        self.img_size = img_size
        self.net = get_net(net_name, img_size)
        if net_name == "isnet_is":
            self.gt_encoder = get_net("isnet_gt", img_size)
            self.gt_encoder.requires_grad_(False)
        else:
            self.gt_encoder = None

    @classmethod
    def try_load(cls, net_name, ckpt_path, map_location=None, img_size=None):
        state_dict = torch.load(ckpt_path, map_location=map_location)
        if "epoch" in state_dict:
            # pytorch-lightning checkpoint
            return cls.load_from_checkpoint(
                ckpt_path, net_name=net_name, img_size=img_size, map_location=map_location
            )
        else:
            model = cls(net_name, img_size)
            if any(k.startswith("net.") for k in state_dict):
                model.load_state_dict(state_dict)
            else:
                model.net.load_state_dict(state_dict)
            return model

    def forward(self, x):
        if isinstance(self.net, (ISNetDIS, ISNetGTEncoder)):
            # [0] = list of 6 side outputs; [-1] = deepest/best segmentation
            # [0] was edge-only with max~0.48; [-1] has max~1.0
            return self.net(x)[0][-1].sigmoid()
        elif isinstance(self.net, U2NET):
            return self.net(x)[0].sigmoid()
        elif isinstance(self.net, MODNet):
            return self.net(x, True)[2]
        elif isinstance(self.net, InSPyReNet):
            return self.net.forward_inference(x)["pred"]
        raise NotImplementedError
