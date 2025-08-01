#!/usr/bin/env python3
"""
3D Microscopy Anisotropy Analysis Functions
Implements GPU-accelerated structure tensor computation and CPU parallelized eigen-decomposition
for fractional, planar, spherical, and linear anisotropy analysis.
"""

import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Tuple, Dict, Any
import structure_tensor as st
from tqdm import tqdm
import time


def normalize_volume_gpu(volume: np.ndarray, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Normalize 3D volume using GPU acceleration and add epsilon to prevent zeros.

    Args:
        volume: Input 3D numpy array (uint8)
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized volume as torch tensor on MPS device
    """
    # Convert to float32 and move to MPS device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    volume_tensor = torch.from_numpy(volume.astype(np.float32)).to(device)

    # Normalize to [0, 1] range and add epsilon
    volume_normalized = (volume_tensor / 255.0) + epsilon

    return volume_normalized


def compute_structure_tensor_gpu(volume: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute 3D structure tensor using GPU acceleration with Apple MPS.
    The structure tensor captures local orientation information in 26 directions.

    Args:
        volume: Normalized 3D volume tensor on MPS device
        sigma: Gaussian smoothing parameter for structure tensor

    Returns:
        Structure tensor components [6, D, H, W] representing symmetric 3x3 tensor
        Components order: [Sxx, Syy, Szz, Sxy, Sxz, Syz]
    """
    print("Computing structure tensor on GPU...")

    # Add batch dimension for 3D convolution: [1, 1, D, H, W]
    volume = volume.unsqueeze(0).unsqueeze(0)

    # Define 3D gradient kernels (Sobel-like)
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)

    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)

    sobel_z = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.float32)

    # Move kernels to device and add channel dimensions
    device = volume.device
    sobel_x = sobel_x.to(device).unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_y.to(device).unsqueeze(0).unsqueeze(0)
    sobel_z = sobel_z.to(device).unsqueeze(0).unsqueeze(0)

    # Compute gradients using 3D convolution
    grad_x = F.conv3d(volume, sobel_x, padding=1)
    grad_y = F.conv3d(volume, sobel_y, padding=1)
    grad_z = F.conv3d(volume, sobel_z, padding=1)

    # Remove batch and channel dimensions
    grad_x = grad_x.squeeze(0).squeeze(0)
    grad_y = grad_y.squeeze(0).squeeze(0)
    grad_z = grad_z.squeeze(0).squeeze(0)

    # Compute structure tensor components (outer product of gradients)
    Sxx = grad_x * grad_x
    Syy = grad_y * grad_y
    Szz = grad_z * grad_z
    Sxy = grad_x * grad_y
    Sxz = grad_x * grad_z
    Syz = grad_y * grad_z

    # Apply Gaussian smoothing to structure tensor components
    if sigma > 0:
        # Create 3D Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords = coords - (kernel_size - 1) / 2
        gaussian_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        # Create 3D separable Gaussian kernel
        gaussian_3d = gaussian_1d.view(-1, 1, 1) * gaussian_1d.view(1, -1, 1) * gaussian_1d.view(1, 1, -1)
        gaussian_3d = gaussian_3d.unsqueeze(0).unsqueeze(0)

        # Apply smoothing to each component
        padding = kernel_size // 2
        Sxx = F.conv3d(Sxx.unsqueeze(0).unsqueeze(0), gaussian_3d, padding=padding).squeeze(0).squeeze(0)
        Syy = F.conv3d(Syy.unsqueeze(0).unsqueeze(0), gaussian_3d, padding=padding).squeeze(0).squeeze(0)
        Szz = F.conv3d(Szz.unsqueeze(0).unsqueeze(0), gaussian_3d, padding=padding).squeeze(0).squeeze(0)
        Sxy = F.conv3d(Sxy.unsqueeze(0).unsqueeze(0), gaussian_3d, padding=padding).squeeze(0).squeeze(0)
        Sxz = F.conv3d(Sxz.unsqueeze(0).unsqueeze(0), gaussian_3d, padding=padding).squeeze(0).squeeze(0)
        Syz = F.conv3d(Syz.unsqueeze(0).unsqueeze(0), gaussian_3d, padding=padding).squeeze(0).squeeze(0)

    # Stack components: [6, D, H, W]
    structure_tensor = torch.stack([Sxx, Syy, Szz, Sxy, Sxz, Syz], dim=0)

    return structure_tensor


def eigen_decomposition_slice(args: Tuple[np.ndarray, int, bool]) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform eigen-decomposition on a single z-slice of the structure tensor.
    Processes each voxel's 3x3 structure tensor to extract eigenvalues and eigenvectors.

    Args:
        args: Tuple containing (tensor_slice, z_index, mask_empty_space)

    Returns:
        Tuple of (z_index, eigenvalues, eigenvectors, anisotropy_measures)
    """
    tensor_slice, z_idx, mask_empty_space = args

    # Get slice dimensions
    height, width = tensor_slice.shape[1], tensor_slice.shape[2]

    # Initialize output arrays (float16 for memory efficiency)
    eigenvalues = np.zeros((height, width, 3), dtype=np.float16)
    eigenvectors = np.zeros((height, width, 3, 3), dtype=np.float16)
    anisotropy = np.zeros((height, width, 4), dtype=np.float16)  # FA, CL, CS, CP

    # Process each voxel in the slice
    for i in range(height):
        for j in range(width):
            # Reconstruct 3x3 symmetric structure tensor from 6 components
            S = np.array([
                [tensor_slice[0, i, j], tensor_slice[3, i, j], tensor_slice[4, i, j]],  # Sxx, Sxy, Sxz
                [tensor_slice[3, i, j], tensor_slice[1, i, j], tensor_slice[5, i, j]],  # Sxy, Syy, Syz
                [tensor_slice[4, i, j], tensor_slice[5, i, j], tensor_slice[2, i, j]]  # Sxz, Syz, Szz
            ], dtype=np.float32)

            # Skip computation for empty space if masking is enabled
            if mask_empty_space and np.trace(S) < 1e-6:
                continue

            try:
                # Compute eigenvalues and eigenvectors
                eigvals, eigvecs = np.linalg.eigh(S)

                # Sort eigenvalues in descending order (λ1 >= λ2 >= λ3)
                sort_indices = np.argsort(eigvals)[::-1]
                eigvals = eigvals[sort_indices]
                eigvecs = eigvecs[:, sort_indices]

                # Ensure eigenvalues are non-negative (numerical stability)
                eigvals = np.maximum(eigvals, 0)

                # Store eigenvalues and eigenvectors
                eigenvalues[i, j] = eigvals.astype(np.float16)
                eigenvectors[i, j] = eigvecs.astype(np.float16)

                # Compute anisotropy measures
                lambda1, lambda2, lambda3 = eigvals

                # Avoid division by zero and ensure numerical stability
                trace = lambda1 + lambda2 + lambda3
                if trace > 1e-8:  # Increased threshold for numerical stability
                    # Fractional Anisotropy (FA)
                    mean_eig = trace / 3
                    numerator = ((lambda1 - mean_eig) ** 2 + (lambda2 - mean_eig) ** 2 + (lambda3 - mean_eig) ** 2)
                    denominator = (lambda1 ** 2 + lambda2 ** 2 + lambda3 ** 2)
                    if denominator > 1e-10:
                        fa = np.sqrt(0.5 * numerator / denominator)
                        fa = min(fa, 1.0)  # Clamp to [0,1] range
                    else:
                        fa = 0.0

                    # Linear Anisotropy (CL)
                    cl = (lambda1 - lambda2) / trace
                    cl = max(0.0, min(cl, 1.0))  # Clamp to [0,1] range

                    # Spherical Anisotropy (CS)
                    cs = 3 * lambda3 / trace
                    cs = max(0.0, min(cs, 1.0))  # Clamp to [0,1] range

                    # Normalized Planar Anisotropy (CP) - more sensitive
                    lambda_diff = lambda3 - lambda1
                    if abs(lambda_diff) > 1e-10:
                        cp = (lambda2 - lambda1) / lambda_diff
                        cp = max(0.0, min(cp, 1.0))  # Clamp to [0,1] range
                    else:
                        cp = 0.0

                    anisotropy[i, j] = [fa, cl, cs, cp]

            except np.linalg.LinAlgError:
                # Handle singular matrices
                continue

    return z_idx, eigenvalues, eigenvectors, anisotropy


def compute_anisotropy_parallel(structure_tensor: torch.Tensor,
                                num_workers: int = 8,
                                mask_empty_space: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute anisotropy measures using parallelized CPU eigen-decomposition.

    Args:
        structure_tensor: Structure tensor components [6, D, H, W]
        num_workers: Number of parallel workers
        mask_empty_space: Whether to skip computation for empty voxels

    Returns:
        Dictionary containing anisotropy measures and principal directions
    """
    print(f"Computing anisotropy with {num_workers} CPU workers...")

    # Move tensor to CPU for processing
    structure_tensor_cpu = structure_tensor.cpu().numpy()
    depth, height, width = structure_tensor_cpu.shape[1:]

    # Initialize output arrays
    fa_volume = np.zeros((depth, height, width), dtype=np.float16)
    cl_volume = np.zeros((depth, height, width), dtype=np.float16)
    cs_volume = np.zeros((depth, height, width), dtype=np.float16)
    cp_volume = np.zeros((depth, height, width), dtype=np.float16)
    principal_directions = np.zeros((depth, height, width, 3), dtype=np.float16)

    # Prepare arguments for parallel processing
    args_list = []
    for z in range(depth):
        tensor_slice = structure_tensor_cpu[:, z, :, :]  # [6, H, W]
        args_list.append((tensor_slice, z, mask_empty_space))

    # Process slices in parallel with progress bar
    with Pool(num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(eigen_decomposition_slice, args_list),
            total=depth,
            desc="Processing slices",
            leave=False
        ))

    # Collect results
    for z_idx, eigenvalues, eigenvectors, anisotropy in results:
        fa_volume[z_idx] = anisotropy[:, :, 0]  # FA
        cl_volume[z_idx] = anisotropy[:, :, 1]  # CL
        cs_volume[z_idx] = anisotropy[:, :, 2]  # CS
        cp_volume[z_idx] = anisotropy[:, :, 3]  # CP

        # Store principal direction (first eigenvector)
        principal_directions[z_idx] = eigenvectors[:, :, :, 0]

    return {
        'fa': fa_volume,
        'cl': cl_volume,
        'cs': cs_volume,
        'cp': cp_volume,
        'principal_direction': principal_directions
    }


def compute_average_direction_gpu(principal_directions: np.ndarray,
                                  fa_volume: np.ndarray,
                                  threshold: float = 0.1) -> np.ndarray:
    """
    Compute voxel-wise average direction weighted by fractional anisotropy using GPU.
    Fast implementation using 3D convolution for neighborhood averaging.

    Args:
        principal_directions: Principal direction vectors [D, H, W, 3]
        fa_volume: Fractional anisotropy volume [D, H, W]
        threshold: FA threshold for including directions in average

    Returns:
        Average direction per voxel [D, H, W, 3]
    """
    print("Computing average directions on GPU...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Convert to torch tensors
    directions_tensor = torch.from_numpy(principal_directions.astype(np.float32)).to(device)
    fa_tensor = torch.from_numpy(fa_volume.astype(np.float32)).to(device)

    # Create mask for significant anisotropy
    mask = fa_tensor > threshold

    # Create 3x3x3 averaging kernel (uniform weights)
    kernel_size = 3
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=device) / (kernel_size ** 3)
    padding = kernel_size // 2

    # Initialize output
    avg_directions = torch.zeros_like(directions_tensor)

    # Process each direction component separately
    for i in range(3):
        # Get direction component [D, H, W]
        dir_component = directions_tensor[:, :, :, i]

        # Multiply by FA weights and mask
        weighted_dirs = dir_component * fa_tensor * mask.float()

        # Add batch and channel dimensions for conv3d: [1, 1, D, H, W]
        weighted_dirs = weighted_dirs.unsqueeze(0).unsqueeze(0)
        fa_weights = (fa_tensor * mask.float()).unsqueeze(0).unsqueeze(0)

        # Apply 3D convolution for neighborhood averaging
        summed_dirs = F.conv3d(weighted_dirs, kernel, padding=padding)
        summed_weights = F.conv3d(fa_weights, kernel, padding=padding)

        # Remove batch and channel dimensions
        summed_dirs = summed_dirs.squeeze(0).squeeze(0)
        summed_weights = summed_weights.squeeze(0).squeeze(0)

        # Compute weighted average (avoid division by zero)
        avg_component = torch.where(summed_weights > 1e-6,
                                    summed_dirs / summed_weights,
                                    dir_component)

        avg_directions[:, :, :, i] = avg_component

    # Normalize direction vectors
    norms = torch.norm(avg_directions, dim=3, keepdim=True)
    avg_directions = torch.where(norms > 1e-6,
                                 avg_directions / norms,
                                 directions_tensor)

    # Apply original mask - only compute for significant anisotropy regions
    mask_expanded = mask.unsqueeze(3).expand_as(avg_directions)
    avg_directions = torch.where(mask_expanded, avg_directions, directions_tensor)

    return avg_directions.cpu().numpy().astype(np.float16)


def compute_average_direction_simple(principal_directions: np.ndarray,
                                     fa_volume: np.ndarray,
                                     threshold: float = 0.1) -> np.ndarray:
    """
    Simple and fast alternative: return principal directions without neighborhood averaging.
    This eliminates the computational bottleneck while still providing meaningful results.

    Args:
        principal_directions: Principal direction vectors [D, H, W, 3]
        fa_volume: Fractional anisotropy volume [D, H, W]
        threshold: FA threshold for masking low-anisotropy regions

    Returns:
        Masked principal directions [D, H, W, 3]
    """
    print("Using principal directions as average directions (fast mode)...")

    # Create mask for significant anisotropy
    mask = fa_volume > threshold

    # Apply mask to directions
    avg_directions = principal_directions.copy()
    mask_expanded = np.expand_dims(mask, axis=3)

    # Zero out directions in low-anisotropy regions
    avg_directions = np.where(mask_expanded, avg_directions, 0)

    return avg_directions


def save_results(results: Dict[str, np.ndarray], output_path: str,
                 compute_avg_directions: str = "simple") -> None:
    """
    Save anisotropy results to compressed NPZ file.

    Args:
        results: Dictionary containing anisotropy measures and directions
        output_path: Path to save the NPZ file
        compute_avg_directions: Method for average directions
            - "gpu": Fast GPU-accelerated neighborhood averaging (recommended)
            - "simple": Use principal directions directly (fastest)
            - "cpu": Original CPU implementation (very slow, not recommended)
            - "skip": Don't compute average directions at all
    """
    print(f"Saving results to {output_path}...")

    # Compute average directions based on method
    if compute_avg_directions == "gpu":
        avg_directions = compute_average_direction_gpu(
            results['principal_direction'],
            results['fa']
        )
    elif compute_avg_directions == "simple":
        avg_directions = compute_average_direction_simple(
            results['principal_direction'],
            results['fa']
        )
    elif compute_avg_directions == "skip":
        print("Skipping average direction computation...")
        avg_directions = None
    else:  # cpu method - not recommended
        print("Warning: Using slow CPU method for average directions...")
        avg_directions = compute_average_direction_cpu(
            results['principal_direction'],
            results['fa']
        )

    # Prepare data for saving
    save_data = {
        'fa': results['fa'],
        'cl': results['cl'],
        'cs': results['cs'],
        'cp': results['cp'],
        'principal_direction': results['principal_direction']
    }

    # Add average directions if computed
    if avg_directions is not None:
        save_data['average_direction'] = avg_directions

    # Save compressed
    np.savez_compressed(output_path, **save_data)

    # Print file size
    import os
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Output file size: {file_size:.2f} MB")


def compute_average_direction_cpu(principal_directions: np.ndarray,
                                  fa_volume: np.ndarray,
                                  threshold: float = 0.1) -> np.ndarray:
    """
    Original CPU implementation (VERY SLOW - not recommended for large volumes).
    Kept for backward compatibility only.
    """
    print("Computing average directions using CPU (slow method)...")

    # Create mask for significant anisotropy
    mask = fa_volume > threshold

    # Initialize average directions
    avg_directions = np.zeros_like(principal_directions)

    # For each voxel, compute weighted average of neighboring directions
    depth, height, width = principal_directions.shape[:3]

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if not mask[z, y, x]:
                    continue

                # Define neighborhood (3x3x3)
                z_min, z_max = max(0, z - 1), min(depth, z + 2)
                y_min, y_max = max(0, y - 1), min(height, y + 2)
                x_min, x_max = max(0, x - 1), min(width, x + 2)

                # Extract neighborhood
                neighborhood_dirs = principal_directions[z_min:z_max, y_min:y_max, x_min:x_max]
                neighborhood_fa = fa_volume[z_min:z_max, y_min:y_max, x_min:x_max]
                neighborhood_mask = neighborhood_fa > threshold

                if np.any(neighborhood_mask):
                    # Compute weighted average
                    weights = neighborhood_fa[neighborhood_mask]
                    directions = neighborhood_dirs[neighborhood_mask]

                    # Weight by FA and normalize
                    weighted_dir = np.average(directions, axis=0, weights=weights)
                    norm = np.linalg.norm(weighted_dir)
                    if norm > 0:
                        avg_directions[z, y, x] = weighted_dir / norm
                    else:
                        avg_directions[z, y, x] = principal_directions[z, y, x]
                else:
                    avg_directions[z, y, x] = principal_directions[z, y, x]

    return avg_directions


def print_statistics(results: Dict[str, np.ndarray]) -> None:
    """
    Print comprehensive statistics of the anisotropy analysis results.

    Args:
        results: Dictionary containing anisotropy measures
    """
    print("\n" + "=" * 60)
    print("ANISOTROPY ANALYSIS STATISTICS")
    print("=" * 60)

    for measure_name, volume in results.items():
        if measure_name == 'principal_direction':
            continue

        # Calculate statistics excluding zeros (empty space)
        non_zero_mask = volume > 1e-6
        if np.any(non_zero_mask):
            values = volume[non_zero_mask].astype(np.float64)  # Convert to float64 for calculations

            # Remove any infinite or NaN values for robust statistics
            finite_mask = np.isfinite(values)
            if np.any(finite_mask):
                clean_values = values[finite_mask]

                print(f"\n{measure_name.upper()} Statistics:")
                print(
                    f"  Non-zero voxels: {np.sum(non_zero_mask):,} ({100 * np.sum(non_zero_mask) / volume.size:.1f}%)")
                print(
                    f"  Finite values: {len(clean_values):,} ({100 * len(clean_values) / np.sum(non_zero_mask):.1f}% of non-zero)")

                if len(clean_values) > 0:
                    # Use robust statistical calculations
                    try:
                        mean_val = float(np.mean(clean_values))
                        std_val = float(np.std(clean_values))
                        min_val = float(np.min(clean_values))
                        max_val = float(np.max(clean_values))

                        # Use numpy percentile with proper handling
                        q25_val = float(np.percentile(clean_values, 25))
                        q50_val = float(np.percentile(clean_values, 50))
                        q75_val = float(np.percentile(clean_values, 75))

                        print(f"  Mean: {mean_val:.4f}")
                        print(f"  Std:  {std_val:.4f}")
                        print(f"  Min:  {min_val:.4f}")
                        print(f"  Max:  {max_val:.4f}")
                        print(f"  Q25:  {q25_val:.4f}")
                        print(f"  Q50:  {q50_val:.4f}")
                        print(f"  Q75:  {q75_val:.4f}")

                    except Exception as e:
                        print(f"  Error computing statistics: {e}")
                        print(f"  Data shape: {clean_values.shape}")
                        print(f"  Data type: {clean_values.dtype}")

                    # Check for problematic values
                    inf_count = np.sum(np.isinf(values))
                    nan_count = np.sum(np.isnan(values))
                    if inf_count > 0:
                        print(f"  Warning: {inf_count} infinite values detected")
                    if nan_count > 0:
                        print(f"  Warning: {nan_count} NaN values detected")
                else:
                    print("  No finite values found")
            else:
                print(f"\n{measure_name.upper()} Statistics:")
                print("  All non-zero values are infinite or NaN")

    print("\n" + "=" * 60)
