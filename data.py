from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan_dataset(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for label, class_name in enumerate(CLASSES):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            canonical_prefix = f"{class_name}_"
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name.startswith(canonical_prefix):
                rows.append({"path": str(path), "label": label, "class_name": class_name})
    return pd.DataFrame(rows)


def make_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby("class_name")
        .size()
        .reindex(CLASSES)
        .reset_index(name="Total Images")
        .rename(columns={"class_name": "Land Cover Class"})
    )
    counts.insert(0, "Class ID", range(1, len(counts) + 1))
    counts["Training"] = (counts["Total Images"] * 0.70).astype(int)
    counts["Validation"] = (counts["Total Images"] * 0.15).astype(int)
    counts["Testing"] = counts["Total Images"] - counts["Training"] - counts["Validation"]
    return counts


def stratified_split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_rgb_transform(input_size: int, train: bool = False):
    ops = []
    if train:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    ops.extend([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
    return transforms.Compose(ops)


class EuroSATDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        input_bands: str = "rgb",
        input_size: int = 224,
        train: bool = False,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.input_bands = input_bands
        self.input_size = input_size
        self.rgb_transform = get_rgb_transform(input_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        if path.suffix.lower() in {".tif", ".tiff"}:
            image = self._read_multispectral(path)
        else:
            image = self._read_rgb(path)

        return image, label

    def _read_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        tensor = self.rgb_transform(image)
        if self.input_bands != "rgb":
            raise ValueError(
                f"Input '{self.input_bands}' membutuhkan file multispectral .tif/.tiff, bukan {path.name}"
            )
        return tensor

    def _read_multispectral(self, path: Path) -> torch.Tensor:
        import rasterio

        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)

        image = select_bands(image, self.input_bands)
        tensor = torch.tensor(image, dtype=torch.float32)
        tensor = resize_tensor_image(tensor, self.input_size)
        return normalize_per_image(tensor)


def select_bands(image: np.ndarray, input_bands: str) -> np.ndarray:
    if input_bands == "rgb":
        return image[[3, 2, 1], :, :]
    if input_bands == "rgb_nir":
        return image[[3, 2, 1, 7], :, :]
    if input_bands == "10m":
        return image[[1, 2, 3, 7], :, :]
    if input_bands == "13bands":
        return image
    if input_bands == "13bands_indices":
        red = image[3]
        green = image[2]
        nir = image[7]
        swir = image[11]
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndwi = (green - nir) / (green + nir + 1e-6)
        ndbi = (swir - nir) / (swir + nir + 1e-6)
        indices = np.stack([ndvi, ndwi, ndbi])
        return np.concatenate([image, indices], axis=0)
    raise ValueError(f"Konfigurasi input_bands tidak dikenal: {input_bands}")


def resize_tensor_image(tensor: torch.Tensor, input_size: int) -> torch.Tensor:
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def normalize_per_image(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1)
    std = flat.std(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    return (tensor - mean) / std


def load_split_csv(csv_path: Path, dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "path" in df.columns and "label" in df.columns:
        normalized = df[["path", "label"]].copy()
        normalized["path"] = normalized["path"].apply(lambda value: remap_path_to_dataset_dir(value, dataset_dir))
        if "class_name" in df.columns:
            normalized["class_name"] = df["class_name"]
        return normalized

    required = {"Filename", "Label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV split harus punya kolom {required} atau path/label: {csv_path}")

    normalized = pd.DataFrame(
        {
            "path": df["Filename"].apply(lambda value: str(dataset_dir / str(value))),
            "label": df["Label"].astype(int),
        }
    )
    if "ClassName" in df.columns:
        normalized["class_name"] = df["ClassName"]
    return normalized


def remap_path_to_dataset_dir(value: str, dataset_dir: Path) -> str:
    path = Path(str(value))
    parts = path.parts
    for idx, part in enumerate(parts):
        if part in CLASSES and idx + 1 < len(parts):
            return str(dataset_dir / part / parts[-1])
    return str(path)
