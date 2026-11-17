"""Pipeline denoising EEG end-to-end berbasis Hilbert RGBA PNG:
sinyal -> Hilbert RGBA PNG -> inferensi 2D Res-UNet -> Hilbert RGBA PNG -> sinyal.

Melatih model khusus untuk artefak EMG dan EOG (split per indeks epoch agar bebas
data leakage) menggunakan panjang sinyal asli dataset tanpa pemotongan, pada
dimensi PNG konstan 32x32x4, dan mengukur kinerja kualitas sinyal serta latensi.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hilbert_png_codec import (
    PngTarget,
    decode_hilbert_png,
    encode_hilbert_png,
    hilbert_grid_to_rgba_array,
    hilbert_indices,
    rgba_array_to_hilbert_grid,
)


class ResBlock2D(nn.Module):
    """Blok residual 2D dengan GroupNorm, Dropout2d ringan, dan aktivasi GELU."""

    def __init__(self, channels: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HilbertUNet(nn.Module):
    """2D Residual U-Net + Bottleneck Self-Attention pada grid Hilbert 32x32."""

    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, base, kernel_size=3, padding=1)
        self.e1 = ResBlock2D(base)
        self.d1 = nn.Conv2d(base, base * 2, kernel_size=2, stride=2)
        self.e2 = ResBlock2D(base * 2)
        self.d2 = nn.Conv2d(base * 2, base * 4, kernel_size=2, stride=2)
        self.e3 = ResBlock2D(base * 4)
        self.d3 = nn.Conv2d(base * 4, base * 4, kernel_size=2, stride=2)

        self.bot = ResBlock2D(base * 4)
        self.attn = nn.MultiheadAttention(base * 4, num_heads=4, batch_first=True)

        self.u3 = nn.ConvTranspose2d(base * 4, base * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(base * 8, base * 4, kernel_size=1), ResBlock2D(base * 4))
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base * 4, base * 2, kernel_size=1), ResBlock2D(base * 2))
        self.u1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base * 2, base, kernel_size=1), ResBlock2D(base))
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.e1(self.stem(x))
        h2 = self.e2(self.d1(h1))
        h3 = self.e3(self.d2(h2))
        b = self.bot(self.d3(h3))

        bsz, ch, h, w = b.shape
        tokens = b.flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(tokens, tokens, tokens, need_weights=False)[0]
        b = tokens.transpose(1, 2).reshape(bsz, ch, h, w)

        p3 = self.dec3(torch.cat([self.u3(b), h3], dim=1))
        p2 = self.dec2(torch.cat([self.u2(p3), h2], dim=1))
        p1 = self.dec1(torch.cat([self.u1(p2), h1], dim=1))
        return self.head(p1)


class HilbertPNGDenoisingPipeline:
    """Pembungkus pipeline lengkap: sinyal -> RGBA PNG -> inferensi -> RGBA PNG -> sinyal."""

    def __init__(self, model: nn.Module, img_size: int = 32, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.img_size = img_size
        self.device = device
        rows, cols = hilbert_indices(img_size)
        self.rows = torch.from_numpy(rows).long()
        self.cols = torch.from_numpy(cols).long()

    @staticmethod
    def signal_to_image(signal: np.ndarray, target: PngTarget, img_size: int = 32) -> tuple[int, int, int]:
        return encode_hilbert_png(signal, target, img_size=img_size)

    @staticmethod
    def image_to_signal(source: PngTarget) -> np.ndarray:
        return decode_hilbert_png(source)

    @torch.no_grad()
    def infer_image(self, in_png: PngTarget, out_png: PngTarget) -> tuple[int, int, int]:
        """Membaca input RGBA PNG, menjalankan inferensi model 2D, dan menyimpan output RGBA PNG."""
        if isinstance(in_png, io.BytesIO):
            in_png.seek(0)
        with Image.open(in_png) as img:
            rgba_in = np.asarray(img, dtype=np.uint8)

        grid_np, n = rgba_array_to_hilbert_grid(rgba_in)
        inp = torch.from_numpy(grid_np).unsqueeze(0).to(self.device)
        pred_grid = self.model(inp).squeeze(0).cpu().numpy()

        rgba_out = hilbert_grid_to_rgba_array(pred_grid, n=n, img_size=self.img_size)
        Image.fromarray(rgba_out, mode="RGBA").save(out_png, format="PNG")
        if isinstance(out_png, io.BytesIO):
            out_png.seek(0)
        return rgba_out.shape

    @torch.no_grad()
    def run_batch_via_png(self, signals: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, float]:
        """Menjalankan seluruh batch melewati encode PNG -> parse PNG -> inferensi -> encode PNG -> decode PNG."""
        n_total, n_len = signals.shape
        out_signals = np.empty_like(signals, dtype=np.float32)

        t0 = time.perf_counter()
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            bsz = end - start

            # 1. Encode tiap sinyal ke PNG bytes dan baca kembali sebagai gambar input model
            grids = np.empty((bsz, 2, self.img_size, self.img_size), dtype=np.float32)
            lengths = np.empty(bsz, dtype=np.int64)
            for i in range(bsz):
                buf = io.BytesIO()
                self.signal_to_image(signals[start + i], buf, img_size=self.img_size)
                with Image.open(buf) as img:
                    rgba = np.asarray(img, dtype=np.uint8)
                grids[i], lengths[i] = rgba_array_to_hilbert_grid(rgba)

            # 2. Inferensi model 2D pada tensor grid Hilbert
            inp = torch.from_numpy(grids).to(self.device)
            pred_grids = self.model(inp).cpu().numpy()

            # 3. Encode output ke PNG bytes lalu decode kembali ke sinyal akhir
            for i in range(bsz):
                rgba_out = hilbert_grid_to_rgba_array(pred_grids[i], n=int(lengths[i]), img_size=self.img_size)
                out_buf = io.BytesIO()
                Image.fromarray(rgba_out, mode="RGBA").save(out_buf, format="PNG")
                out_buf.seek(0)
                out_signals[start + i] = self.image_to_signal(out_buf)

        elapsed = time.perf_counter() - t0
        return out_signals, elapsed


def compute_metrics(clean: np.ndarray, est: np.ndarray) -> dict[str, float]:
    """Menghitung metrik standar EEG denoising: RMSE, RRMSE_t, RRMSE_s, CC, dan SNR (dB)."""
    diff = est - clean
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    norm_clean = np.linalg.norm(clean, axis=1) + 1e-8
    norm_diff = np.linalg.norm(diff, axis=1)
    rrmse_t = float(np.mean(norm_diff / norm_clean))

    fft_clean = np.abs(np.fft.rfft(clean, axis=1))
    fft_est = np.abs(np.fft.rfft(est, axis=1))
    rrmse_s = float(np.mean(np.linalg.norm(fft_est - fft_clean, axis=1) / (np.linalg.norm(fft_clean, axis=1) + 1e-8)))

    c_centered = clean - clean.mean(axis=1, keepdims=True)
    e_centered = est - est.mean(axis=1, keepdims=True)
    cc_vec = np.sum(c_centered * e_centered, axis=1) / (
        np.linalg.norm(c_centered, axis=1) * np.linalg.norm(e_centered, axis=1) + 1e-8
    )
    cc = float(np.mean(cc_vec))

    snr_db = float(np.mean(20.0 * np.log10(norm_clean / (norm_diff + 1e-8))))
    return {
        "rmse": rmse,
        "rrmse_t": rrmse_t,
        "rrmse_s": rrmse_s,
        "cc": cc,
        "snr_db": snr_db,
    }


def _signals_to_grid_batch(
    sig_batch: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    n_active: int,
    img_size: int = 32,
) -> torch.Tensor:
    bsz = sig_batch.shape[0]
    grid = torch.zeros(bsz, 2, img_size, img_size, dtype=sig_batch.dtype, device=sig_batch.device)
    grid[:, 0, rows[:n_active], cols[:n_active]] = sig_batch[:, :n_active]
    grid[:, 1, rows[:n_active], cols[:n_active]] = 1.0
    return grid


def load_splits(artifact_type: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Memuat dataset dan membagi berdasarkan indeks epoch (80/10/10) tanpa leakage."""
    clean = np.load(f"Dataset/{artifact_type}/clean_eeg_all_epochs.npy")
    syn = np.load(f"Dataset/{artifact_type}/linear_synthetic.npy", allow_pickle=True).item()
    snrs = sorted(syn.keys())

    n_epochs = clean.shape[0]
    n_train = int(0.8 * n_epochs)
    n_val = int(0.1 * n_epochs)

    def _gather(sl: slice) -> tuple[np.ndarray, np.ndarray]:
        xs = [syn[s][sl] for s in snrs]
        ys = [clean[sl] for _ in snrs]
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    splits = {
        "train": _gather(slice(0, n_train)),
        "val": _gather(slice(n_train, n_train + n_val)),
        "test": _gather(slice(n_train + n_val, n_epochs)),
    }
    test_by_snr = {
        int(s): (syn[s][n_train + n_val:], clean[n_train + n_val:])
        for s in snrs
    }
    return splits, test_by_snr


def train_model(
    artifact_type: str,
    ckpt_path: Path,
    epochs: int = 5,
    batch_size: int = 256,
    base_channels: int = 24,
    seed: int = 42,
) -> tuple[HilbertUNet, dict[int, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Melatih HilbertUNet pada panjang sinyal asli dataset dan menyimpan checkpoint terbaik."""
    torch.manual_seed(seed)

    splits, test_by_snr = load_splits(artifact_type)
    x_tr, y_tr = splits["train"]
    x_va, y_va = splits["val"]
    signal_len = int(x_tr.shape[1])

    rows_np, cols_np = hilbert_indices(32)
    rows = torch.from_numpy(rows_np).long()
    cols = torch.from_numpy(cols_np).long()

    model = HilbertUNet(base=base_channels)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    x_va_t = torch.from_numpy(x_va)

    best_rmse = float("inf")
    best_metrics: dict[str, float] = {}
    t_start = time.perf_counter()

    print(
        f"\n=== Training {artifact_type} HilbertUNet "
        f"(train={len(x_tr)}, val={len(x_va)}, len={signal_len}, params={sum(p.numel() for p in model.parameters())}) ==="
    )
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            inp = _signals_to_grid_batch(xb, rows, cols, n_active=signal_len)
            pred = model(inp)[:, 0, rows[:signal_len], cols[:signal_len]]

            mse_l = F.mse_loss(pred, yb)
            spec_l = F.l1_loss(
                torch.fft.rfft(pred, dim=-1, norm="ortho").abs(),
                torch.fft.rfft(yb, dim=-1, norm="ortho").abs(),
            )
            loss = mse_l + 0.8 * spec_l

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += float(loss.item())

        sched.step()
        model.eval()
        with torch.no_grad():
            val_preds = []
            for vs in range(0, len(x_va_t), 512):
                vb = x_va_t[vs:vs + 512]
                vinp = _signals_to_grid_batch(vb, rows, cols, n_active=signal_len)
                val_preds.append(model(vinp)[:, 0, rows[:signal_len], cols[:signal_len]].numpy())
            vp = np.concatenate(val_preds, axis=0)
            m = compute_metrics(y_va, vp)

        elapsed = time.perf_counter() - t_start
        print(
            f"  Epoch {ep}/{epochs} ({elapsed:.1f}s) | loss={ep_loss/len(loader):.4f} "
            f"| val_rmse={m['rmse']:.4f}, rrmse_t={m['rrmse_t']:.4f}, rrmse_s={m['rrmse_s']:.4f}, cc={m['cc']:.4f}, snr={m['snr_db']:.2f} dB"
        )
        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_metrics = m
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model, test_by_snr, best_metrics


def benchmark_disk_latency(pipeline: HilbertPNGDenoisingPipeline, signals: np.ndarray, tmp_dir: Path, n_runs: int = 50) -> dict[str, float]:
    """Mengukur latensi aktual per tahap pada file PNG fisik di disk (batch=1) dan cek integritas PNG."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_png = tmp_dir / "_lat_in.png"
    out_png = tmp_dir / "_lat_out.png"

    t_enc, t_inf, t_dec = [], [], []
    max_container_diff = 0.0

    for i in range(n_runs):
        sig = signals[i]
        t0 = time.perf_counter()
        pipeline.signal_to_image(sig, in_png, img_size=32)
        t1 = time.perf_counter()
        pipeline.infer_image(in_png, out_png)
        t2 = time.perf_counter()
        rec = pipeline.image_to_signal(out_png)
        t3 = time.perf_counter()

        # Verifikasi lossless input PNG -> decode balik
        sig_check = pipeline.image_to_signal(in_png)
        max_container_diff = max(max_container_diff, float(np.max(np.abs(sig - sig_check))))
        assert rec.shape == sig.shape

        t_enc.append((t1 - t0) * 1000.0)
        t_inf.append((t2 - t1) * 1000.0)
        t_dec.append((t3 - t2) * 1000.0)

    in_png.unlink(missing_ok=True)
    out_png.unlink(missing_ok=True)
    total_ms = float(np.mean(t_enc) + np.mean(t_inf) + np.mean(t_dec))
    return {
        "n_runs": n_runs,
        "encode_ms": float(np.mean(t_enc)),
        "infer_ms": float(np.mean(t_inf)),
        "decode_ms": float(np.mean(t_dec)),
        "total_ms_per_sample": total_ms,
        "throughput_samples_per_sec": 1000.0 / total_ms,
        "max_png_codec_diff": max_container_diff,
    }


def evaluate_artifact_pipeline(
    artifact_type: str,
    model: HilbertUNet,
    test_by_snr: dict[int, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> dict[str, object]:
    """Evaluasi end-to-end pipeline Hilbert RGBA PNG pada seluruh set uji per SNR (panjang asli)."""
    pipeline = HilbertPNGDenoisingPipeline(model, img_size=32, device="cpu")
    snr_results: dict[str, dict[str, float]] = {}

    all_noisy, all_clean, all_denoised = [], [], []
    total_png_time = 0.0

    print(f"\n=== Evaluasi End-to-End Pipeline {artifact_type} (Test Set melalui Encode/Decode PNG) ===")
    print("SNR_in | RMSE_noisy -> RMSE_den | RRMSE_t_den | RRMSE_s_den | CC_noisy -> CC_den | SNR_out (Delta)")
    for snr, (noisy_arr, clean_arr) in test_by_snr.items():
        denoised_arr, dt = pipeline.run_batch_via_png(noisy_arr, batch_size=256)
        total_png_time += dt

        m_noisy = compute_metrics(clean_arr, noisy_arr)
        m_den = compute_metrics(clean_arr, denoised_arr)
        delta_snr = m_den["snr_db"] - m_noisy["snr_db"]

        snr_results[str(snr)] = {
            "rmse_noisy": m_noisy["rmse"],
            "rmse_denoised": m_den["rmse"],
            "rrmse_t_noisy": m_noisy["rrmse_t"],
            "rrmse_t_denoised": m_den["rrmse_t"],
            "rrmse_s_noisy": m_noisy["rrmse_s"],
            "rrmse_s_denoised": m_den["rrmse_s"],
            "cc_noisy": m_noisy["cc"],
            "cc_denoised": m_den["cc"],
            "snr_in_db": m_noisy["snr_db"],
            "snr_out_db": m_den["snr_db"],
            "delta_snr_db": delta_snr,
        }
        print(
            f"  {snr:3d}  | {m_noisy['rmse']:.4f} -> {m_den['rmse']:.4f}     "
            f"| {m_den['rrmse_t']:.4f}      | {m_den['rrmse_s']:.4f}      "
            f"| {m_noisy['cc']:.4f} -> {m_den['cc']:.4f}   | {m_den['snr_db']:5.2f} dB (+{delta_snr:.2f} dB)"
        )

        all_noisy.append(noisy_arr)
        all_clean.append(clean_arr)
        all_denoised.append(denoised_arr)

    cat_noisy = np.concatenate(all_noisy, axis=0)
    cat_clean = np.concatenate(all_clean, axis=0)
    cat_den = np.concatenate(all_denoised, axis=0)
    ov_noisy = compute_metrics(cat_clean, cat_noisy)
    ov_den = compute_metrics(cat_clean, cat_den)
    overall = {
        "n_test_samples": int(len(cat_clean)),
        "rmse_noisy": ov_noisy["rmse"],
        "rmse_denoised": ov_den["rmse"],
        "rrmse_t_noisy": ov_noisy["rrmse_t"],
        "rrmse_t_denoised": ov_den["rrmse_t"],
        "rrmse_s_noisy": ov_noisy["rrmse_s"],
        "rrmse_s_denoised": ov_den["rrmse_s"],
        "cc_noisy": ov_noisy["cc"],
        "cc_denoised": ov_den["cc"],
        "snr_in_db": ov_noisy["snr_db"],
        "snr_out_db": ov_den["snr_db"],
        "delta_snr_db": ov_den["snr_db"] - ov_noisy["snr_db"],
        "batched_png_pipeline_sec": total_png_time,
    }
    print(
        f"  AVG  | {overall['rmse_noisy']:.4f} -> {overall['rmse_denoised']:.4f}     "
        f"| {overall['rrmse_t_denoised']:.4f}      | {overall['rrmse_s_denoised']:.4f}      "
        f"| {overall['cc_noisy']:.4f} -> {overall['cc_denoised']:.4f}   | {overall['snr_out_db']:5.2f} dB (+{overall['delta_snr_db']:.2f} dB)"
    )

    # Simpan contoh PNG fisik input/output ke disk menggunakan panjang sinyal asli (512)
    tag = artifact_type.lower()
    noisy_ex, _ = test_by_snr[-5]
    in_512 = out_dir / f"{tag}_snr-5_len512_input.png"
    out_512 = out_dir / f"{tag}_snr-5_len512_denoised.png"
    shape_in = pipeline.signal_to_image(noisy_ex[0], in_512, img_size=32)
    shape_out = pipeline.infer_image(in_512, out_512)
    rec_512 = pipeline.image_to_signal(out_512)

    # Ukur latensi disk aktual
    latency = benchmark_disk_latency(pipeline, test_by_snr[-5][0], tmp_dir=out_dir, n_runs=50)
    print(
        f"  Disk Latency (bs=1): encode={latency['encode_ms']:.2f} ms, infer={latency['infer_ms']:.2f} ms, "
        f"decode={latency['decode_ms']:.2f} ms | total={latency['total_ms_per_sample']:.2f} ms "
        f"({latency['throughput_samples_per_sec']:.1f} samples/s) | codec_diff={latency['max_png_codec_diff']:.1e}"
    )

    return {
        "overall": overall,
        "by_snr": snr_results,
        "latency_disk_bs1": latency,
        "sample_files": {
            "full512_in_png": str(in_512),
            "full512_out_png": str(out_512),
            "full512_shape_in": list(shape_in),
            "full512_shape_out": list(shape_out),
            "full512_rec_len": int(rec_512.size),
        },
    }


def main() -> None:
    out_dir = Path("pipeline_outputs")
    ckpt_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    results = {}
    for artifact in ("EOG", "EMG"):
        ckpt = ckpt_dir / f"{artifact.lower()}_hilbert_unet.pt"
        model, test_by_snr, best_val = train_model(
            artifact_type=artifact,
            ckpt_path=ckpt,
            epochs=5,
            batch_size=256,
            base_channels=24,
        )
        eval_res = evaluate_artifact_pipeline(artifact, model, test_by_snr, out_dir=out_dir)
        results[artifact] = {
            "checkpoint": str(ckpt),
            "best_val": best_val,
            "test_evaluation": eval_res,
        }

    report_path = out_dir / "benchmark_results.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nLaporan lengkap tersimpan di: {report_path}")


if __name__ == "__main__":
    main()
