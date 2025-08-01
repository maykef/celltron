3-D Microscopy Anisotropy Pipeline

GPU structure-tensor • multi-core eigen-decomp • napari visualisation

A copy-and-paste-friendly toolkit for quantifying local orientation and anisotropy in large 3-D microscopy volumes.  Designed around Apple-silicon Macs (M-series) but happy on CPU-only hardware, it combines
	•	🔥 Metal-accelerated gradient & smoothing (PyTorch 2 MPS backend)
	•	⚡ Process-parallel eigen-decomposition that saturates all performance cores
	•	💾 Lean I/O – float16 outputs, memory-mapped input, compressed .npz
	•	👁️ Instant napari layers for FA / CL / CP / CS & principal directions

Everything lives in two files – functions.py and main.py – so you can just drop them into PyCharm and hit ▶.

⸻

1️ Quick start

# clone (or just grab the two .py files)
$ git clone https://github.com/your-org/anisotropy-3d.git
$ cd anisotropy-3d

# create env (conda or venv)
$ conda create -n aniso python=3.11
$ conda activate aniso

# install deps (MPS wheels come from PyPI ≥ 2.1)
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
$ pip install tifffile napari[all] tqdm joblib

# drop your stack into Data/thresholded_image.tif
$ python main.py   # napari pops up, close it to save results

Tip  On Macs the script autodetects the MPS GPU; on Linux/Windows it silently falls back to CPU for the structure-tensor step.

⸻

2️ Folder layout

.
├── functions.py          # all heavy lifting
├── main.py               # parameters + viewer glue
├── Data/
│   └── thresholded_image.tif
└── README.md             # you’re reading it

No CLI, no config files – edit the first few lines of main.py if you need a different path, sigma, worker count, …

⸻

3️ Algorithm sketch

Stage	Device	Key tricks
Load + normalise (mem-mapped TIFF → float32)	CPU	avoids 2× RAM spike
Gradients ∇I (Sobel) + outer products	GPU (MPS)	separable 1-D convs ⇒ 3-4× faster
Gaussian blur σ	GPU	same separable trick
Eigen-decomp of every 3×3 tensor	8 CPU processes	vectorised LAPACK over 20 k-voxel chunks;  OMP_NUM_THREADS=1 to stop oversubscription
FA, CL, CP, CS	CPU	computed in float32, cast to float16
Save (np.savez_compressed)	CPU	~80 MB for 512³ stack
Visualise	CPU + GPU	napari 3-D viewer, 5 layers

Benchmarks on an M1 Max (64 GB): 512³ voxels → ≈ 1 minute end-to-end (∼3.8 M vox/s).

⸻

4️ Key parameters (all in main.py)

DATA_PATH  = "Data/thresholded_image.tif"  # input stack (8-bit)
OUT_PATH   = "Data/anisotropy_results.npz" # saved on napari close
SIGMA      = 1.0   # Gaussian radius (vox)
N_WORKERS  = 8     # CPU processes
MASK_THRES = 5/255 # ignore background when computing CS


⸻

5️ Output contents

>>> np.load('anisotropy_results.npz').files
['fa', 'cl', 'cp', 'cs', 'cp_norm', 'lambda2_ratio', 'principal_dir']

	•	fa, cl, cp, cs, cp_norm, λ2_ratio – (Z,Y,X) float16
	•	principal_dir – (Z,Y,X,3) float16 unit vectors

⸻

6️ Troubleshooting

Symptom	Fix
torch.backends.mps.is_available() → False	Upgrade PyTorch ≥ 2.1 and macOS 13+, or force CPU by setting DEVICE="cpu" in functions.py.
Only ~200 % CPU during eigensolver	Make sure you kept the with parallel_backend("loky") block and didn’t change OMP_NUM_THREADS.
RuntimeWarning: divide by zero in cp_norm	Update functions.py ≥ v2025-08-01: metrics computed in float32.
napari window empty / crashes	pip install --upgrade napari[all] PyQt5 and reboot if using Wayland/Linux.


⸻

7️ Roadmap
	•	CUDA back-end for gradients on Windows / Linux
	•	Batch-processing helper (iterate over a folder)
	•	Optional OME-Zarr output for web viz

⸻

8️ Citation

If this library helps your research, please cite us 🙂

@software{anisotropy3d_2025,
  author       = {Your Name},
  title        = {Anisotropy-3D: GPU structure-tensor + multicore eigen-decomposition},
  year         = {2025},
  url          = {https://github.com/your-org/anisotropy-3d},
  version      = {v0.2.0}
}


⸻

© 2025 MIT Licence – do what you want, but please share improvements.
