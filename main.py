#!/usr/bin/env python3
"""
3D Microscopy Anisotropy Analysis - Main Script
Processes 3D microscopy volumes to compute anisotropy measures using GPU-accelerated
structure tensor computation and CPU parallelized eigen-decomposition.
"""

import numpy as np
import torch
from tifffile import imread
import napari
from functions import (
    normalize_volume_gpu,
    compute_structure_tensor_gpu,
    compute_anisotropy_parallel,
    save_results,
    print_statistics
)
import time
from multiprocessing import cpu_count
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """Main processing pipeline for 3D anisotropy analysis."""

    # =============================================================================
    # PARAMETERS CONFIGURATION
    # =============================================================================

    # Input/Output Configuration
    INPUT_PATH = "Data/thresholded_image.tif"
    OUTPUT_PATH = "Data/anisotropy_results.npz"

    # Processing Parameters
    NUM_WORKERS = min(8, cpu_count())  # Use up to 8 CPU cores or available cores
    SIGMA = 1.0  # Gaussian smoothing for structure tensor
    EPSILON = 1e-6  # Small value to prevent zeros
    MASK_EMPTY_SPACE = True  # Skip computation for empty voxels (CS anisotropy)

    # Average Direction Computation Method
    AVG_DIRECTION_METHOD = "gpu"  # Options: "gpu", "simple", "skip", "cpu"
    # "gpu": Fast GPU-accelerated neighborhood averaging (recommended for quality)
    # "simple": Use principal directions directly (fastest, good quality)
    # "skip": Don't compute average directions (fastest, saves space)
    # "cpu": Original slow CPU method (not recommended for large volumes)

    # Display Parameters
    NAPARI_VIEWER = True  # Launch napari viewer for results
    SHOW_DIRECTION_VECTORS = False  # Show principal direction vectors (can be cluttering)

    print("3D MICROSCOPY ANISOTROPY ANALYSIS")
    print("=" * 50)
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"CPU workers: {NUM_WORKERS}")
    print(f"GPU device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"Structure tensor sigma: {SIGMA}")
    print(f"Mask empty space: {MASK_EMPTY_SPACE}")
    print(f"Average direction method: {AVG_DIRECTION_METHOD}")
    print(f"Show direction vectors: {SHOW_DIRECTION_VECTORS}")
    print("=" * 50)

    # =============================================================================
    # LOAD AND VALIDATE INPUT DATA
    # =============================================================================

    print("\n1. Loading 3D volume...")
    start_time = time.time()

    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # Load TIFF volume
    try:
        volume = imread(INPUT_PATH)
        print(f"   Loaded volume shape: {volume.shape}")
        print(f"   Data type: {volume.dtype}")
        print(f"   Value range: [{volume.min()}, {volume.max()}]")
        print(f"   Memory usage: {volume.nbytes / 1024 ** 2:.1f} MB")
    except Exception as e:
        raise RuntimeError(f"Failed to load volume: {e}")

    # Validate volume properties
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D")

    if volume.dtype != np.uint8:
        print(f"   Warning: Converting from {volume.dtype} to uint8")
        volume = volume.astype(np.uint8)

    load_time = time.time() - start_time
    print(f"   Loading completed in {load_time:.2f} seconds")

    # =============================================================================
    # STEP 1: VOLUME NORMALIZATION (GPU)
    # =============================================================================

    print("\n2. Normalizing volume on GPU...")
    norm_start = time.time()

    # Normalize volume and move to GPU
    volume_normalized = normalize_volume_gpu(volume, epsilon=EPSILON)
    print(f"   Normalized volume range: [{volume_normalized.min():.6f}, {volume_normalized.max():.6f}]")
    print(
        f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB" if torch.cuda.is_available() else "   Using MPS backend")

    norm_time = time.time() - norm_start
    print(f"   Normalization completed in {norm_time:.2f} seconds")

    # =============================================================================
    # STEP 2: STRUCTURE TENSOR COMPUTATION (GPU)
    # =============================================================================

    print("\n3. Computing structure tensor on GPU...")
    tensor_start = time.time()

    # Compute structure tensor using GPU acceleration
    structure_tensor = compute_structure_tensor_gpu(volume_normalized, sigma=SIGMA)
    print(f"   Structure tensor shape: {structure_tensor.shape}")
    print(f"   Structure tensor components: [Sxx, Syy, Szz, Sxy, Sxz, Syz]")

    # Print tensor statistics
    for i, component in enumerate(['Sxx', 'Syy', 'Szz', 'Sxy', 'Sxz', 'Syz']):
        tensor_component = structure_tensor[i]
        print(f"   {component}: mean={tensor_component.mean():.6f}, std={tensor_component.std():.6f}")

    tensor_time = time.time() - tensor_start
    print(f"   Structure tensor computation completed in {tensor_time:.2f} seconds")

    # =============================================================================
    # STEP 3: EIGEN-DECOMPOSITION AND ANISOTROPY (CPU PARALLEL)
    # =============================================================================

    print(f"\n4. Computing anisotropy with {NUM_WORKERS} CPU workers...")
    eigen_start = time.time()

    # Compute anisotropy measures using parallel CPU processing
    results = compute_anisotropy_parallel(
        structure_tensor=structure_tensor,
        num_workers=NUM_WORKERS,
        mask_empty_space=MASK_EMPTY_SPACE
    )

    eigen_time = time.time() - eigen_start
    print(f"   Eigen-decomposition completed in {eigen_time:.2f} seconds")

    # =============================================================================
    # STEP 4: RESULTS STATISTICS AND VALIDATION
    # =============================================================================

    print("\n5. Computing statistics...")
    stats_start = time.time()

    # Print comprehensive statistics
    print_statistics(results)

    # Validate results
    for measure_name, volume_data in results.items():
        if measure_name == 'principal_direction':
            continue

        # Check for NaN or infinite values
        if np.any(np.isnan(volume_data)) or np.any(np.isinf(volume_data)):
            print(f"   Warning: {measure_name} contains NaN or infinite values")

        # Check value ranges
        if measure_name in ['fa', 'cl', 'cs', 'cp']:
            if np.any(volume_data < 0) or np.any(volume_data > 1):
                print(f"   Warning: {measure_name} values outside [0,1] range")

    stats_time = time.time() - stats_start
    print(f"   Statistics computed in {stats_time:.2f} seconds")

    # =============================================================================
    # STEP 5: SAVE RESULTS
    # =============================================================================

    print(f"\n6. Saving results to {OUTPUT_PATH}...")
    save_start = time.time()

    # Save results to compressed NPZ file
    save_results(results, OUTPUT_PATH, compute_avg_directions=AVG_DIRECTION_METHOD)

    save_time = time.time() - save_start
    print(f"   Results saved in {save_time:.2f} seconds")

    # =============================================================================
    # STEP 6: LAUNCH NAPARI VIEWER
    # =============================================================================

    if NAPARI_VIEWER:
        print("\n7. Launching napari viewer...")
        viewer_start = time.time()

        # Create napari viewer
        viewer = napari.Viewer(title="3D Anisotropy Analysis Results")

        # Add original volume
        viewer.add_image(
            volume,
            name="Original Volume",
            colormap="gray",
            opacity=0.5
        )

        # Add anisotropy measures as separate layers
        viewer.add_image(
            results['fa'],
            name="Fractional Anisotropy (FA)",
            colormap="viridis",
            opacity=0.8
        )

        viewer.add_image(
            results['cl'],
            name="Linear Anisotropy (CL)",
            colormap="plasma",
            opacity=0.7
        )

        viewer.add_image(
            results['cs'],
            name="Spherical Anisotropy (CS)",
            colormap="inferno",
            opacity=0.7
        )

        viewer.add_image(
            results['cp'],
            name="Planar Anisotropy (CP)",
            colormap="magma",
            opacity=0.7
        )

        # Add principal directions as vectors (optional - can be cluttering)
        if SHOW_DIRECTION_VECTORS:
            print("   Adding direction vectors...")
            downsample_factor = 8  # Increased downsampling for better performance

            # Create downsampled coordinate grids
            z_indices = np.arange(0, results['principal_direction'].shape[0], downsample_factor)
            y_indices = np.arange(0, results['principal_direction'].shape[1], downsample_factor)
            x_indices = np.arange(0, results['principal_direction'].shape[2], downsample_factor)

            # Get coordinates for downsampled grid
            coords_list = []
            directions_list = []

            for z in z_indices:
                for y in y_indices:
                    for x in x_indices:
                        # Get FA value to filter low-anisotropy regions
                        fa_val = results['fa'][z, y, x]
                        if fa_val > 0.2:  # Only show vectors where FA > 0.2
                            # Get direction vector
                            direction = results['principal_direction'][z, y, x]

                            # Check if direction is non-zero
                            if np.linalg.norm(direction) > 0.1:
                                coords_list.append([z, y, x])
                                # Scale direction for visualization
                                directions_list.append(direction * 30)  # Scale factor for visibility

            if len(coords_list) > 0:
                # Convert to numpy arrays
                coords = np.array(coords_list, dtype=np.float32)
                directions = np.array(directions_list, dtype=np.float32)

                # Create vector data in napari format: [N, 2, D] where N is number of vectors
                # Each vector has [start_point, end_point] in D dimensions
                vector_data = np.stack([coords, coords + directions], axis=1)

                print(f"   Adding {len(coords)} direction vectors")

                # Add vectors to viewer
                viewer.add_vectors(
                    data=vector_data,
                    name="Principal Directions",
                    edge_color="red",
                    edge_width=2,
                    length=1.0,
                    opacity=0.7
                )
            else:
                print("   No significant direction vectors to display")
        else:
            print("   Skipping direction vectors (SHOW_DIRECTION_VECTORS=False)")

        viewer_time = time.time() - viewer_start
        print(f"   Napari viewer launched in {viewer_time:.2f} seconds")

        # =============================================================================
        # TOTAL PROCESSING TIME
        # =============================================================================

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"  - Loading: {load_time:.2f}s ({100 * load_time / total_time:.1f}%)")
        print(f"  - Normalization: {norm_time:.2f}s ({100 * norm_time / total_time:.1f}%)")
        print(f"  - Structure tensor: {tensor_time:.2f}s ({100 * tensor_time / total_time:.1f}%)")
        print(f"  - Eigen-decomposition: {eigen_time:.2f}s ({100 * eigen_time / total_time:.1f}%)")
        print(f"  - Statistics: {stats_time:.2f}s ({100 * stats_time / total_time:.1f}%)")
        print(f"  - Saving: {save_time:.2f}s ({100 * save_time / total_time:.1f}%)")
        print(f"  - Viewer setup: {viewer_time:.2f}s ({100 * viewer_time / total_time:.1f}%)")
        print(f"\nProcessing rate: {volume.size / total_time:.0f} voxels/second")
        print(f"Memory efficiency: {volume.nbytes / 1024 ** 2 / total_time:.1f} MB/second")

        print(f"\nResults saved to: {OUTPUT_PATH}")
        print("Napari viewer is open. Close the viewer window when done.")
        print("The results will be saved automatically when you close napari.")

        # Start napari event loop
        napari.run()

        print("\nNapari viewer closed. Analysis complete!")

    else:
        # =============================================================================
        # TOTAL PROCESSING TIME (WITHOUT NAPARI)
        # =============================================================================

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"  - Loading: {load_time:.2f}s ({100 * load_time / total_time:.1f}%)")
        print(f"  - Normalization: {norm_time:.2f}s ({100 * norm_time / total_time:.1f}%)")
        print(f"  - Structure tensor: {tensor_time:.2f}s ({100 * tensor_time / total_time:.1f}%)")
        print(f"  - Eigen-decomposition: {eigen_time:.2f}s ({100 * eigen_time / total_time:.1f}%)")
        print(f"  - Statistics: {stats_time:.2f}s ({100 * stats_time / total_time:.1f}%)")
        print(f"  - Saving: {save_time:.2f}s ({100 * save_time / total_time:.1f}%)")
        print(f"\nProcessing rate: {volume.size / total_time:.0f} voxels/second")
        print(f"Memory efficiency: {volume.nbytes / 1024 ** 2 / total_time:.1f} MB/second")

        print(f"\nResults saved to: {OUTPUT_PATH}")
        print("Analysis complete!")


if __name__ == "__main__":
    """
    Entry point for the 3D anisotropy analysis pipeline.

    Requirements:
    - PyTorch with MPS support for Apple Silicon
    - tifffile for TIFF I/O
    - napari for 3D visualization
    - numpy, multiprocessing, tqdm
    - structure_tensor library

    The script processes a 3D microscopy volume to compute:
    - Fractional Anisotropy (FA)
    - Linear Anisotropy (CL) 
    - Spherical Anisotropy (CS)
    - Planar Anisotropy (CP)
    - Principal directions and average directions

    Processing pipeline:
    1. Load and normalize 3D volume
    2. Compute structure tensor on GPU (26-directional analysis)
    3. Parallel CPU eigen-decomposition for anisotropy measures
    4. Save results to compressed NPZ file
    5. Visualize results in napari
    """

    # Check system requirements
    print("Checking system requirements...")

    # Check PyTorch MPS availability
    if torch.backends.mps.is_available():
        print("✓ PyTorch MPS backend available")
    else:
        print("⚠ PyTorch MPS backend not available, using CPU")

    # Check multiprocessing capability
    available_cpus = cpu_count()
    print(f"✓ Available CPU cores: {available_cpus}")

    # Check memory
    import psutil

    available_memory = psutil.virtual_memory().available / 1024 ** 3
    print(f"✓ Available RAM: {available_memory:.1f} GB")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback

        traceback.print_exc()
