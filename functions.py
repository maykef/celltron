"""
functions.py – GPU-accelerated structure-tensor utilities + CPU-parallel
                             eigen-decomposition for 3-D microscopy volumes

Author  : you
Updated : 2025-08-01

The module is intentionally **self-contained** and free of CLI/argparse so it can
be copy-pasted into PyCharm and imported by a companion *main.py* script.

Key design decisions
--------------------
* **MPS / Apple-silicon first** – all convolutions & filtering run on the GPU
  via PyTorch’s Metal backend.  If MPS is unavailable the code silently
  falls back to CPU.
* **Eigen-decomposition on CPU processes** – `np.linalg.eigh` is highly
  optimised in Apple’s Accelerate framework but isn’t exposed on GPU.  A
  process-based `joblib` pool lets us use all performance cores without GIL
  contention.
* **Memory-efficient** – float32 for intermediate volumes, float16 for output
  metrics, compressed `.npz` saving (handled in *main.py*).
* **No unused imports** – everything below is referenced.

External dependencies
---------------------
```
pip install numpy torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install joblib tqdm tifffile
```
(When running on Apple-silicon you’ll get the MPS build of PyTorch via the
binary above.)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import tifffile as tfi
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 0.  DEVICE SET-UP
# -----------------------------------------------------------------------------
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
DTYPE: torch.dtype = torch.float32  # adequate precision, leaner than float64
EPS: float = 1e-6                  # avoids divide-by-zero in anisotropy formulas


# -----------------------------------------------------------------------------
# 1.  I/O HELPERS
# -----------------------------------------------------------------------------

def load_volume(path: str | os.PathLike) -> np.ndarray:
    """Loads an 8-bit TIFF stack as *memory-mapped* float32 array in the range 0-1.

    Using `tifffile.memmap` keeps RAM usage negligible until slices are first
    accessed – handy for large volumes.
    """
    vol_mm = tfi.memmap(path)
    vol_np = np.asarray(vol_mm, dtype=np.float32) / 255.0  # 0-1 normalisation
    return vol_np + EPS  # shift to avoid exact zeros


# -----------------------------------------------------------------------------
# 2.  GPU STRUCTURE-TENSOR
# -----------------------------------------------------------------------------
# 2-a. 1-D Sobel coefficients and derivative kernel (pre-allocated on DEVICE)
_SOBEL_1D = torch.tensor([1.0, 2.0, 1.0], dtype=DTYPE, device=DEVICE) / 4.0
_DERIV_1D = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE, device=DEVICE) / 2.0


def _gradient_3d(vol_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separable 3-D Sobel gradient on the GPU.

    Parameters
    ----------
    vol_t : torch.Tensor of shape [1, 1, Z, Y, X]

    Returns
    -------
    gx, gy, gz : each same shape as *vol_t*
    """
    # ∂/∂x ---------------------------------------------------------------------
    gx = F.conv3d(vol_t, _DERIV_1D[None, None, :, None, None], padding=(1, 0, 0))
    gx = F.conv3d(gx,     _SOBEL_1D[None, None, None, :, None], padding=(0, 1, 0))
    gx = F.conv3d(gx,     _SOBEL_1D[None, None, None, None, :], padding=(0, 0, 1))

    # ∂/∂y ---------------------------------------------------------------------
    gy = F.conv3d(vol_t, _DERIV_1D[None, None, None, :, None], padding=(0, 1, 0))
    gy = F.conv3d(gy,     _SOBEL_1D[None, None, :, None, None], padding=(1, 0, 0))
    gy = F.conv3d(gy,     _SOBEL_1D[None, None, None, None, :], padding=(0, 0, 1))

    # ∂/∂z ---------------------------------------------------------------------
    gz = F.conv3d(vol_t, _DERIV_1D[None, None, None, None, :], padding=(0, 0, 1))
    gz = F.conv3d(gz,     _SOBEL_1D[None, None, :, None, None], padding=(1, 0, 0))
    gz = F.conv3d(gz,     _SOBEL_1D[None, None, None, :, None], padding=(0, 1, 0))

    return gx, gy, gz


def compute_structure_tensor(vol_np: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, ...]:
    """Computes the six unique components of the 3×3 structure tensor.

    Parameters
    ----------
    vol_np : ndarray  (Z, Y, X) float32
    sigma  : Gaussian smoothing radius (voxels). Values <≈0.3 disable smoothing.

    Returns
    -------
    (Jxx, Jyy, Jzz, Jxy, Jxz, Jyz) – each `np.float32`, shape (Z, Y, X)
    """
    # ---- send volume to device ------------------------------------------------
    vol_t = torch.from_numpy(vol_np).to(DEVICE, non_blocking=True)[None, None]

    # ---- gradient fields ------------------------------------------------------
    gx, gy, gz = _gradient_3d(vol_t)

    # ---- outer products -------------------------------------------------------
    Jxx, Jyy, Jzz = gx * gx, gy * gy, gz * gz
    Jxy, Jxz, Jyz = gx * gy, gx * gz, gy * gz

    # ---- optional Gaussian smoothing -----------------------------------------
    if sigma > 0.3:
        size = int(2 * 3 * sigma + 1)
        ax   = torch.arange(size, device=DEVICE) - size // 2
        kernel1d = torch.exp(-(ax**2) / (2 * sigma**2))
        kernel1d = (kernel1d / kernel1d.sum()).to(DTYPE)

        def gblur(t: torch.Tensor) -> torch.Tensor:
            t = F.conv3d(t, kernel1d[None, None, :, None, None], padding=(size//2, 0, 0))
            t = F.conv3d(t, kernel1d[None, None, None, :, None], padding=(0, size//2, 0))
            t = F.conv3d(t, kernel1d[None, None, None, None, :], padding=(0, 0, size//2))
            return t

        Jxx, Jyy, Jzz, Jxy, Jxz, Jyz = map(gblur, (Jxx, Jyy, Jzz, Jxy, Jxz, Jyz))

    # ---- back to CPU numpy ----------------------------------------------------
    comps = [t.squeeze().cpu().numpy().astype(np.float32) for t in (Jxx, Jyy, Jzz, Jxy, Jxz, Jyz)]
    return tuple(comps)


# -----------------------------------------------------------------------------
# 3.  CPU-PARALLEL EIGEN-DECOMPOSITION
# -----------------------------------------------------------------------------

def _eig_chunk(chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised eigen-decomposition of *N* 3×3 symmetric matrices stored
    as (N,6) [Jxx, Jyy, Jzz, Jxy, Jxz, Jyz].

    Returns
    -------
    eigenvalues : (N,3) float16  (ordered λ1≥λ2≥λ3)
    eigenvecs   : (N,3) float16  (principal directions)
    """
    if chunk.size == 0:
        # safety for empty slices
        return np.empty((0, 3), np.float16), np.empty((0, 3), np.float16)

    Jxx, Jyy, Jzz, Jxy, Jxz, Jyz = chunk.T
    M = np.stack([
        Jxx, Jxy, Jxz,
        Jxy, Jyy, Jyz,
        Jxz, Jyz, Jzz
    ], axis=-1).reshape(-1, 3, 3)

    vals, vecs = np.linalg.eigh(M)  # Accelerate/BLAS under the hood
    order = vals.argsort(axis=-1)[:, ::-1]  # descending λ1≥λ2≥λ3
    idx = np.arange(vals.shape[0])[:, None]

    vals_sorted = vals[idx, order].astype(np.float16)
    dirs = vecs[idx, :, order][:, :, 0].astype(np.float16)  # principal dir → λ1
    return vals_sorted, dirs


def eigen_parallel(
    components: Tuple[np.ndarray, ...],
    mask: np.ndarray,
    n_jobs: int = 8,
    chunk_size: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parallel eigen-decomposition over masked voxels.

    Parameters
    ----------
    components : tuple of 6 ndarrays from `compute_structure_tensor`
    mask       : bool ndarray of same shape – voxels to process
    n_jobs     : number of CPU processes (use physical cores, not threads)
    chunk_size : voxels per joblib task – tune for cache/overhead balance

    Returns
    -------
    eigvals : (N,3) float16   eigenvalues λ1≥λ2≥λ3
    eigvecs : (N,3) float16   principal directions (unit vectors)
    """
    # Prevent each spawned process from internally multi-threading BLAS –
    # otherwise you end up with n_jobs × OMP_THREADS threads and slower code.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    # Flatten 6 component volumes to shape (total_voxels, 6) --------------------
    flat6 = np.stack(components, axis=-1).reshape(-1, 6)
    voxels = flat6[mask.ravel()]  # (N,6)

    # Chunk generator (lazy) to keep memory footprint low -----------------------
    def gen_chunks():
        for i in range(0, voxels.shape[0], chunk_size):
            yield voxels[i : i + chunk_size]

    with parallel_backend("loky", n_jobs=n_jobs):
        results = Parallel(verbose=0)(
            delayed(_eig_chunk)(c)
            for c in tqdm(gen_chunks(), desc="Eigen-decomp", unit="chunk")
        )

    eigvals = np.vstack([r[0] for r in results])
    eigvecs = np.vstack([r[1] for r in results])
    return eigvals, eigvecs


# -----------------------------------------------------------------------------
# 4.  ANISOTROPY METRICS
# -----------------------------------------------------------------------------

def anisotropy_metrics(eigvals: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes standard anisotropy scalars from eigenvalues.

    Notes
    -----
    * All formulas follow Westin et al. (2002) conventions with λ1≥λ2≥λ3.
    * Empty-space voxels should be excluded *before* calling this function;
      they would otherwise skew spherical anisotropy.
    """
    λ1, λ2, λ3 = eigvals.T
    λ̄ = (λ1 + λ2 + λ3) / 3.0 + EPS

    # Fractional anisotropy (FA) -----------------------------------------------
    fa = np.sqrt(1.5 * ((λ1 - λ̄) ** 2 + (λ2 - λ̄) ** 2 + (λ3 - λ̄) ** 2) /
                 (λ1 ** 2 + λ2 ** 2 + λ3 ** 2 + EPS))

    # Westin linear, planar, spherical -----------------------------------------
    cl = (λ1 - λ2) / (λ1 + EPS)
    cp = (λ2 - λ3) / (λ1 + EPS)
    cs = (3.0 * λ3) / (λ1 + λ2 + λ3 + EPS)

    # Requested extra ratios ----------------------------------------------------
    cp_norm = (λ2 - λ1) / (λ3 - λ1 + EPS)   # normalised planar
    λ2_ratio = λ2 / (λ3 + EPS)

    return {
        "fa":        fa.astype(np.float16),
        "cl":        cl.astype(np.float16),
        "cp":        cp.astype(np.float16),
        "cs":        cs.astype(np.float16),
        "cp_norm":   cp_norm.astype(np.float16),
        "lambda2_ratio": λ2_ratio.astype(np.float16),
    }
