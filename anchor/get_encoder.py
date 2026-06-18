import json
import logging
import os

import timm
import torch
from timm.data import create_transform, resolve_data_config
from timm.layers import SwiGLUPacked
from torchvision import transforms

_DEFAULT_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_eval_transforms(img_resize=224, center_crop=False):
    eval_transform = []
    if img_resize > 0:
        eval_transform.append(transforms.Resize(img_resize))
        if center_crop:
            eval_transform.append(transforms.CenterCrop(img_resize))

    eval_transform.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(eval_transform)


def to_anchor_embedding(output):
    """Convert ANCHOR token output [B, 261, 1280] to [B, 2560] embedding."""
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


def _load_anchor_from_config(assets_dir, checkpoint):
    config_path = os.path.join(assets_dir, "config.json")
    ckpt_path = os.path.join(assets_dir, checkpoint)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing weights: {ckpt_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    model_args = dict(cfg["model_args"])
    mlp_layer = SwiGLUPacked if model_args.pop("mlp_layer", None) == "SwiGLUPacked" else None
    act_layer = torch.nn.SiLU if model_args.pop("act_layer", None) == "SiLU" else None

    model = timm.create_model(
        cfg["architecture"],
        pretrained=False,
        mlp_layer=mlp_layer,
        act_layer=act_layer,
        **model_args,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

    eval_transform = create_transform(**resolve_data_config(cfg["pretrained_cfg"], model=model))
    return model, eval_transform, missing_keys, unexpected_keys


def get_encoder(
    enc_name="anchor",
    checkpoint="pytorch_model.bin",
    img_resize=224,
    center_crop=True,
    test_batch=0,
    device=None,
    assets_dir=None,
):
    """
    Load ANCHOR image encoder and preprocessing transform.

    Args:
        enc_name: Encoder preset name. Currently supports ``'anchor'``.
        checkpoint: Weight filename inside ``assets_dir``.
        assets_dir: Directory containing ``config.json`` and ``pytorch_model.bin``.
            Defaults to ``../weights`` relative to this file.

    Returns:
        model: timm ViT backbone. Forward pass returns token tensor [B, 261, 1280].
        eval_transform: torchvision/timm transform for 224x224 RGB tiles.
    """
    enc_name_presets = {
        "anchor": ("anchor", "pytorch_model.bin"),
    }
    if enc_name in enc_name_presets:
        enc_name, checkpoint = enc_name_presets[enc_name]

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if assets_dir is None:
        assets_dir = _DEFAULT_ASSETS_DIR
    assets_dir = os.path.abspath(assets_dir)

    if enc_name == "anchor":
        model, eval_transform, missing_keys, unexpected_keys = _load_anchor_from_config(
            assets_dir, checkpoint
        )
    else:
        return None, None

    logging.info("Missing Keys: %s", missing_keys)
    logging.info("Unexpected Keys: %s", unexpected_keys)
    logging.info(str(model))

    model.eval()
    model.to(device)

    logging.info("Transform Type: %s", eval_transform)
    if test_batch:
        imgs = torch.rand((2, 3, img_resize, img_resize), device=device)
        with torch.no_grad():
            tokens = model(imgs)
            features = to_anchor_embedding(tokens)
        logging.info(
            "Test batch successful, token shape: %s, embedding dim: %s",
            tuple(tokens.shape),
            features.size(1),
        )
        del imgs, tokens, features

    return model, eval_transform
