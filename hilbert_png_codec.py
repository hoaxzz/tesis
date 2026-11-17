"""Codec lossless untuk konversi sinyal EEG 1D panjang dinamis <-> Hilbert RGBA PNG.

Setiap sampel float32 dikemas menjadi 4 byte RGBA little-endian sepanjang jalur
Hilbert 2D (grid berukuran tetap img_size x img_size, default 32x32), sementara
piksel Hilbert terakhir menyimpan panjang sinyal asli N (uint32).
"""

from functools import lru_cache
import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PngTarget = Union[str, Path, io.BytesIO]


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _d2xy(n: int, d: int) -> tuple[int, int]:
    t, x, y, s = d, 0, 0, 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


@lru_cache(maxsize=8)
def hilbert_indices(img_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Mengembalikan koordinat (rows, cols) kurva Hilbert untuk grid pangkat 2."""
    if img_size <= 0 or (img_size & (img_size - 1)) != 0:
        raise ValueError(f"img_size harus pangkat 2 positif, bukan {img_size}.")
    coords = [_d2xy(img_size, d) for d in range(img_size * img_size)]
    xs, ys = zip(*coords)
    return np.asarray(ys, dtype=np.int64), np.asarray(xs, dtype=np.int64)


def signal_to_rgba_array(signal: np.ndarray, img_size: int = 32) -> np.ndarray:
    """Mengemas sinyal float32 panjang N <= img_size^2 - 1 ke array RGBA uint8."""
    sig = np.ascontiguousarray(signal, dtype="<f4").reshape(-1)
    n = sig.size
    max_cap = img_size * img_size - 1
    if n == 0 or n > max_cap:
        raise ValueError(f"Panjang sinyal {n} harus berada di rentang [1, {max_cap}].")

    rows, cols = hilbert_indices(img_size)
    rgba = np.zeros((img_size, img_size, 4), dtype=np.uint8)
    rgba[rows[:n], cols[:n]] = sig.view(np.uint8).reshape(n, 4)
    rgba[rows[-1], cols[-1]] = np.array([n], dtype="<u4").view(np.uint8)
    return rgba


def rgba_array_to_signal(rgba: np.ndarray) -> np.ndarray:
    """Mengekstrak kembali sinyal float32 panjang N secara bit-exact dari array RGBA."""
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.shape[0] != rgba.shape[1]:
        raise ValueError(f"Bentuk array RGBA tidak valid: {rgba.shape}")
    img_size = rgba.shape[0]
    rows, cols = hilbert_indices(img_size)

    n = int(np.ascontiguousarray(rgba[rows[-1], cols[-1]]).view("<u4")[0])
    max_cap = img_size * img_size - 1
    if n <= 0 or n > max_cap:
        raise ValueError(f"Metadata panjang sinyal tidak valid di piksel Hilbert terakhir: {n}")

    sig_bytes = np.ascontiguousarray(rgba[rows[:n], cols[:n]])
    return sig_bytes.view("<f4").reshape(n)


def rgba_array_to_hilbert_grid(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """Mengubah array RGBA menjadi tensor grid Hilbert 2-channel (signal, active_mask)."""
    img_size = rgba.shape[0]
    rows, cols = hilbert_indices(img_size)
    sig = rgba_array_to_signal(rgba)
    n = sig.size

    grid = np.zeros((2, img_size, img_size), dtype=np.float32)
    grid[0, rows[:n], cols[:n]] = sig
    grid[1, rows[:n], cols[:n]] = 1.0
    return grid, n


def hilbert_grid_to_rgba_array(grid: np.ndarray, n: int, img_size: int = 32) -> np.ndarray:
    """Mengemas prediksi grid Hilbert (H, W) atau (1, H, W) + panjang N menjadi RGBA uint8."""
    plane = grid[0] if grid.ndim == 3 else grid
    rows, cols = hilbert_indices(img_size)
    sig = np.ascontiguousarray(plane[rows[:n], cols[:n]], dtype="<f4")
    return signal_to_rgba_array(sig, img_size=img_size)


def encode_hilbert_png(signal: np.ndarray, target: PngTarget, img_size: int = 32) -> tuple[int, int, int]:
    """Menyimpan sinyal 1D ke file/buffer Hilbert RGBA PNG dan mengembalikan shape gambar."""
    rgba = signal_to_rgba_array(signal, img_size=img_size)
    Image.fromarray(rgba, mode="RGBA").save(target, format="PNG")
    if isinstance(target, io.BytesIO):
        target.seek(0)
    return rgba.shape


def decode_hilbert_png(source: PngTarget) -> np.ndarray:
    """Membaca file/buffer Hilbert RGBA PNG dan mengembalikan sinyal float32 asli."""
    if isinstance(source, io.BytesIO):
        source.seek(0)
    with Image.open(source) as img:
        rgba = np.asarray(img, dtype=np.uint8)
    return rgba_array_to_signal(rgba)
