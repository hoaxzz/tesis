"""Demonstrasi dan verifikasi dua teknik konversi sinyal EEG ke gambar
berdimensi konstan yang dapat direkonstruksi kembali ke sinyal asli.

Teknik 1: Hilbert Curve + Bit-Packing RGBA PNG (100% bit-exact, 0 error).
Teknik 2: Complex STFT (Real + Imaginary + Length) dengan Zero-Padding ke N_max.
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image


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


def hilbert_indices(img_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Menghasilkan koordinat (row, col) kurva Hilbert untuk grid persegi pangkat 2."""
    coords = [_d2xy(img_size, d) for d in range(img_size * img_size)]
    xs, ys = zip(*coords)
    return np.asarray(ys, dtype=np.int64), np.asarray(xs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Teknik 1: Hilbert Curve + Bit-Packing RGBA PNG (Bit-Exact)
# ---------------------------------------------------------------------------

def encode_hilbert_png(signal: np.ndarray, out_path: Path, img_size: int = 32) -> tuple[int, int, int]:
    """Mengubah sinyal float32 panjang dinamis (<= img_size^2 - 1) menjadi RGBA PNG konstan."""
    sig = np.ascontiguousarray(signal, dtype="<f4")
    n = sig.size
    max_cap = img_size * img_size - 1
    if n == 0 or n > max_cap:
        raise ValueError(f"Panjang sinyal {n} harus berada di rentang [1, {max_cap}].")

    rows, cols = hilbert_indices(img_size)
    rgba = np.zeros((img_size, img_size, 4), dtype=np.uint8)
    rgba[rows[:n], cols[:n]] = sig.view(np.uint8).reshape(n, 4)
    rgba[rows[-1], cols[-1]] = np.array([n], dtype="<u4").view(np.uint8)

    Image.fromarray(rgba, mode="RGBA").save(out_path, format="PNG")
    return rgba.shape


def decode_hilbert_png(png_path: Path) -> np.ndarray:
    """Mengembalikan file RGBA PNG menjadi sinyal float32 asli secara bit-exact."""
    rgba = np.asarray(Image.open(png_path), dtype=np.uint8)
    img_size = rgba.shape[0]
    rows, cols = hilbert_indices(img_size)

    n = int(np.ascontiguousarray(rgba[rows[-1], cols[-1]]).view("<u4")[0])
    sig_bytes = np.ascontiguousarray(rgba[rows[:n], cols[:n]])
    return sig_bytes.view("<f4").reshape(n)


# ---------------------------------------------------------------------------
# Teknik 2: Invertible Complex STFT (Real + Imag + Length) dengan Padding N_max
# ---------------------------------------------------------------------------

def encode_complex_stft(
    signal: np.ndarray,
    out_path: Path,
    n_max: int = 1024,
    n_fft: int = 64,
    hop_length: int = 16,
) -> tuple[int, int, int]:
    """Mengubah sinyal panjang dinamis (<= n_max) menjadi tensor citra 3-channel konstan."""
    sig = np.ascontiguousarray(signal, dtype=np.float32)
    n = sig.size
    if n == 0 or n > n_max:
        raise ValueError(f"Panjang sinyal {n} harus berada di rentang [1, {n_max}].")

    padded = torch.zeros(n_max, dtype=torch.float32)
    padded[:n] = torch.from_numpy(sig)

    window = torch.hann_window(n_fft, dtype=torch.float32)
    stft_c = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )

    length_plane = torch.full_like(stft_c.real, fill_value=float(n) / float(n_max))
    img_tensor = torch.stack([stft_c.real, stft_c.imag, length_plane], dim=0).numpy()
    np.save(out_path, img_tensor)
    return img_tensor.shape


def decode_complex_stft(
    npy_path: Path,
    n_max: int = 1024,
    n_fft: int = 64,
    hop_length: int = 16,
) -> np.ndarray:
    """Mengembalikan tensor citra Complex STFT menjadi sinyal float32 asli via ISTFT."""
    img_tensor = torch.from_numpy(np.load(npy_path))
    stft_c = torch.complex(img_tensor[0], img_tensor[1])
    n = int(round(float(img_tensor[2, 0, 0].item()) * n_max))

    window = torch.hann_window(n_fft, dtype=torch.float32)
    reconstructed = torch.istft(
        stft_c,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        length=n_max,
    )
    return reconstructed[:n].numpy()


def evaluate_reconstruction(name: str, orig: np.ndarray, rec: np.ndarray) -> dict[str, object]:
    """Menghitung metrik kemiripan sinyal asli vs hasil rekonstruksi."""
    diff = orig - rec
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(orig, rec)[0, 1])
    return {
        "case": name,
        "len_orig": orig.size,
        "len_rec": rec.size,
        "bit_exact": bool(np.array_equal(orig, rec)),
        "allclose_1e5": bool(np.allclose(orig, rec, atol=1e-5, rtol=1e-5)),
        "max_abs_err": max_abs,
        "rmse": rmse,
        "pearson_r": corr,
    }


def main() -> None:
    out_dir = Path("demo_outputs")
    out_dir.mkdir(exist_ok=True)

    emg_dict = np.load("Dataset/EMG/linear_synthetic.npy", allow_pickle=True).item()
    eog_dict = np.load("Dataset/EOG/linear_synthetic.npy", allow_pickle=True).item()

    # 2 sampel terkontaminasi utama + 1 uji panjang dinamis (dipotong 300 sampel)
    samples = [
        ("EMG_SNR-7_idx0_full512", emg_dict[-7][0]),
        ("EOG_SNR-5_idx0_full512", eog_dict[-5][0]),
        ("EOG_SNR-5_idx0_dyn300", eog_dict[-5][0][:300]),
    ]

    print("=== HASIL UJI KONVERSI & REKONSTRUKSI SINYAL TERKONTAMINASI ===")
    for label, sig in samples:
        png_path = out_dir / f"{label}_hilbert.png"
        stft_path = out_dir / f"{label}_stft.npy"

        shape_t1 = encode_hilbert_png(sig, png_path, img_size=32)
        rec_t1 = decode_hilbert_png(png_path)
        res_t1 = evaluate_reconstruction(f"{label} [Hilbert PNG]", sig, rec_t1)

        shape_t2 = encode_complex_stft(sig, stft_path, n_max=1024, n_fft=64, hop_length=16)
        rec_t2 = decode_complex_stft(stft_path, n_max=1024, n_fft=64, hop_length=16)
        res_t2 = evaluate_reconstruction(f"{label} [Complex STFT]", sig, rec_t2)

        print(f"\nSample: {label} | shape asli: {sig.shape} | dtype: {sig.dtype}")
        print(f"  Teknik 1 (Hilbert RGBA PNG) -> image shape: {shape_t1}, file: {png_path} ({png_path.stat().st_size} bytes)")
        print(f"    len_rec={res_t1['len_rec']}, bit_exact={res_t1['bit_exact']}, max_abs_err={res_t1['max_abs_err']:.3e}, rmse={res_t1['rmse']:.3e}, r={res_t1['pearson_r']:.10f}")
        print(f"  Teknik 2 (Complex STFT)     -> image shape: {shape_t2}, file: {stft_path} ({stft_path.stat().st_size} bytes)")
        print(f"    len_rec={res_t2['len_rec']}, bit_exact={res_t2['bit_exact']}, allclose_1e5={res_t2['allclose_1e5']}, max_abs_err={res_t2['max_abs_err']:.3e}, rmse={res_t2['rmse']:.3e}, r={res_t2['pearson_r']:.10f}")


if __name__ == "__main__":
    main()
