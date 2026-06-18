import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from anchor.get_encoder import get_encoder, to_anchor_embedding

_DEFAULT_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


class RoiDataset(Dataset):
    """Image list dataset. ``img_csv`` must have a ``filename`` column."""

    def __init__(self, img_csv, transform):
        super().__init__()
        self.transform = transform
        if isinstance(img_csv, pd.DataFrame):
            self.image_paths = img_csv["filename"].tolist()
        else:
            self.image_paths = list(img_csv)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


def get_features(
    img_csv,
    batch_size=1,
    device=None,
    assets_dir=None,
    num_workers=0,
):
    """
    Extract ANCHOR embeddings for a list of tile images.

    Args:
        img_csv: ``pandas.DataFrame`` with a ``filename`` column, or a list of image paths.
        batch_size: DataLoader batch size.
        device: Torch device. Defaults to CUDA if available.
        assets_dir: Directory containing ``config.json`` and ``pytorch_model.bin``.

    Returns:
        numpy.ndarray of shape [N, 2560].
    """
    model, transform = get_encoder(
        device=device,
        assets_dir=assets_dir or _DEFAULT_ASSETS_DIR,
    )
    dataset = RoiDataset(img_csv, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = next(model.parameters()).device
    features = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            tokens = model(batch)
            embedding = to_anchor_embedding(tokens)
            features.append(embedding.cpu())

    return torch.cat(features, dim=0).numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract ANCHOR features from a CSV image list.")
    parser.add_argument("--csv", default="./test_list.csv", help="CSV with a filename column")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default=None, help="Optional .npy output path")
    parser.add_argument("--assets-dir", default=_DEFAULT_ASSETS_DIR, help="Path to weights directory")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    img_csv = pd.read_csv(args.csv)
    features = get_features(
        img_csv,
        batch_size=args.batch_size,
        device=device,
        assets_dir=args.assets_dir,
    )
    print(f"features: shape={features.shape}, mean={features.mean():.6f}")

    if args.output:
        import numpy as np

        np.save(args.output, features)
        print(f"saved: {args.output}")
