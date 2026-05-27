"""Encode simulator / file observations for vision LLM planners."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional, Tuple

import numpy as np

_IMAGE_KEYS = (
    "agentview_image",
    "image",
    "agentview_rgb",
    "rgb",
)


def find_image_array(obs_dict: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Return HxWxC uint8 RGB from the latest frame in obs_dict, or None."""
    if not obs_dict:
        return None
    for key in _IMAGE_KEYS:
        if key not in obs_dict:
            continue
        arr = _tensor_to_numpy(obs_dict[key])
        if arr is None:
            continue
        rgb = _to_hwc_uint8(arr)
        if rgb is not None:
            return rgb
    return None


def image_file_to_obs_dict(path: str, key: str = "agentview_image") -> Dict[str, Any]:
    from PIL import Image

    img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return {key: img[np.newaxis, np.newaxis, ...]}


def encode_rgb_jpeg_data_url(rgb: np.ndarray, quality: int = 85) -> str:
    from PIL import Image

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _tensor_to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return x


def _to_hwc_uint8(arr: np.ndarray) -> Optional[np.ndarray]:
    """Accept (H,W,C), (C,H,W), (T,C,H,W), (B,T,C,H,W), normalized or 0-255."""
    if arr.size == 0:
        return None

    if arr.ndim == 5:
        arr = arr[0, -1]
    elif arr.ndim == 4:
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr[-1]
        else:
            arr = arr[-1]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        pass

    if arr.ndim != 3:
        return None

    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0 and arr.min() >= -1.5:
            arr = (arr * 127.5 + 127.5).clip(0, 255)
        elif arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
    arr = arr.astype(np.uint8)
    if arr.shape[-1] != 3:
        return None
    return arr


def obs_summary(obs_dict: Optional[Dict[str, Any]]) -> str:
    """Short text hint when no image is available."""
    if not obs_dict:
        return "No observation provided."
    parts = []
    for k, v in obs_dict.items():
        if "image" in k.lower() or "rgb" in k.lower():
            continue
        if hasattr(v, "shape"):
            parts.append(f"{k}: shape={tuple(v.shape)}")
    return "; ".join(parts) if parts else "Observation dict has no proprio keys."
