3D Microscopy Anisotropy Analysis

A high-performance, copy-and-paste-friendly toolkit for 3-D microscopy anisotropy analysis.
It couples Metal-accelerated structure-tensor gradients with a vectorised, multi-process eigen-solver that fully saturates modern Apple-silicon CPUs.  End-to-end, a 512³ stack processes in ≈ 1 min on an M1 Max.

⸻

🚀 Key Features

Category	Highlights
🔥 GPU Acceleration	Separable Sobel + Gaussian blur run on the MPS backend (3–4× faster than CPU)
⚡ CPU Parallelism	Eigen-decomposition chunked & vectorised across 8 processes; no GIL bottleneck, no thread oversubscription
📊 Full Anisotropy Suite	FA, CL, CP, CS + CPnorm & λ₂/λ₃ ratios with guaranteed eigenvalue ordering
💾 Memory Smart	Float32 computation → Float16 storage; memory-mapped TIFF; compressed .npz output
📈 Clean Progress	Single tqdm bar—workers stay silent
🎯 Smart Masking	Background voxels auto-excluded so CS isn’t inflated
👁️ Interactive Viz	One-click napari viewer with five colour-mapped layers + principal-direction vectors
🪄 Zero-CLI	Two files (functions.py, main.py); edit paths in main.py, hit ▶ in PyCharm


⸻

🧬 Scientific Background
	1.	Structure tensor captures 3-D local gradients in 26 directions (faces, edges, corners).
	2.	Eigen-decomposition of each 3×3 tensor yields principal directions (λ₁≥λ₂≥λ₃).
	3.	Anisotropy metrics translate eigenvalues into biologically interpretable scalars:
	•	FA – fractional anisotropy, overall coherence
	•	CL / CP / CS – linear, planar, spherical (Westin 2002)
	•	CPnorm & λ₂/λ₃ – more sensitive planar descriptors

⸻

📋 Requirements

Hardware
	•	macOS 13+ recommended – Apple Silicon (M-series) for GPU path
	•	≥ 8 GB RAM (≥ 32 GB for 1024³ volumes)

Python deps

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # MPS wheel on macOS
pip install tifffile napari[all] tqdm joblib structure-tensor


⸻

⚡ Quick Installation

# grab the two scripts
wget https://raw.githubusercontent.com/your-org/anisotropy-3d/main/functions.py
wget https://raw.githubusercontent.com/your-org/anisotropy-3d/main/main.py

# put your stack at Data/thresholded_image.tif and run
python main.py

Close the napari window to write anisotropy_results.npz.

⸻

🎯 Usage & Config

Edit the header of main.py:

DATA_PATH  = "Data/thresholded_image.tif"   # input 8-bit TIFF
OUT_PATH   = "Data/anisotropy_results.npz"  # written on napari close
SIGMA      = 1.0   # Gaussian radius (voxels)
N_WORKERS  = 8     # CPU processes (perf cores)
MASK_THRES = 5/255 # ignore near-black voxels

Run:

python main.py


⸻

📊 Output

Compressed .npz contains fa, cl, cp, cs, cp_norm, lambda2_ratio (Z × Y × X float16) and principal_dir (Z × Y × X × 3 float16).  Typical size for 512³ ≈ 80 MB.

⸻

📈 Benchmarks (M1 Max 32-GPU-core, 8 CPU perf-core)

Volume	Time	CPU util	Peak RAM
256³	9 s	730 %	1.4 GB
512³	58 s	760 %	5.3 GB
768³	2 min 15 s	780 %	11 GB


⸻

🔧 Advanced Tweaks
	•	chunk_size in functions.eigen_parallel controls per-process workload (20 000 voxels is sweet-spot).
	•	Set DEVICE = "cpu" in functions.py to benchmark pure-CPU fallback.
	•	Change colormaps / down-sample vector field directly in main.py.

⸻

🛠️ Troubleshooting

Symptom	Fix
MPS available: False	Upgrade PyTorch ≥ 2.1; macOS ≥ 13; otherwise runs CPU-only
Only 200 % CPU	Make sure parallel_backend("loky") block is intact; don’t switch to threads
RuntimeWarning: divide by zero	You’re on an older commit – update functions.py (metrics now computed in float32)


⸻

📄 License

MIT – free to use, modify, and share.  If you publish results, please cite:

@software{anisotropy3d_2025,
  author  = {Your Name},
  title   = {3-D Microscopy Anisotropy Toolkit},
  year    = {2025},
  url     = {https://github.com/your-org/anisotropy-3d},
  version = {v0.2.1}
}


⸻

Made with ❤️ for the microscopy community – accelerating 3-D tissue analysis with modern GPU computing.
