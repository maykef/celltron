#!/usr/bin/env python3
"""
Setup script for 3D Microscopy Anisotropy Analysis
GPU-accelerated structure tensor computation with CPU parallelized eigen-decomposition
Now with OME-Zarr + NGFF multiscale output support
"""

from setuptools import setup, find_packages
import os
import sys


# Read the README file for long description
def read_readme():
    """Read README.md file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "3D Microscopy Anisotropy Analysis with GPU acceleration, CPU parallelization, and OME-Zarr output"


# Read requirements from requirements.txt if it exists
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


# Define base requirements
BASE_REQUIREMENTS = [
    'numpy>=1.21.0',
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'tifffile>=2021.7.2',
    'napari[all]>=0.4.18',
    'tqdm>=4.62.0',
    'psutil>=5.8.0',
    'joblib>=1.1.0',
    # OME-Zarr and NGFF dependencies
    'ome-zarr>=0.8.0',
    'zarr>=2.12.0',
    'dask[array]>=2022.8.0',
    'scikit-image>=0.19.0',
    'fsspec>=2022.7.1',
]

# Platform-specific requirements
PLATFORM_REQUIREMENTS = {
    'darwin': [  # macOS
        'torch>=2.0.0',  # Ensure MPS support
    ],
    'linux': [
        'torch>=2.0.0',
    ],
    'win32': [
        'torch>=2.0.0',
    ]
}

# Development requirements
DEV_REQUIREMENTS = [
    'pytest>=6.0.0',
    'pytest-cov>=2.12.0',
    'black>=21.0.0',
    'flake8>=3.9.0',
    'isort>=5.9.0',
    'mypy>=0.910',
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
]

# Documentation requirements
DOC_REQUIREMENTS = [
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    'sphinxcontrib-napoleon>=0.7',
    'matplotlib>=3.4.0',  # For documentation plots
]

# Cloud storage requirements (optional)
CLOUD_REQUIREMENTS = [
    's3fs>=2022.7.1',      # AWS S3 support
    'gcsfs>=2022.7.1',     # Google Cloud Storage
    'adlfs>=2022.7.1',     # Azure Data Lake Storage
]

# High-performance requirements (optional)
PERFORMANCE_REQUIREMENTS = [
    'numba>=0.56.0',       # JIT compilation for faster processing
    'cupy>=11.0.0',        # CUDA-accelerated NumPy (NVIDIA GPUs)
    'cucim>=22.08.0',      # CUDA-accelerated image processing
]

# Get platform-specific requirements
current_platform = sys.platform
platform_reqs = PLATFORM_REQUIREMENTS.get(current_platform, [])

# Combine requirements
install_requires = BASE_REQUIREMENTS + platform_reqs

# Read additional requirements from file if available
file_requirements = read_requirements()
if file_requirements:
    install_requires.extend(file_requirements)

# Remove duplicates while preserving order
seen = set()
install_requires = [x for x in install_requires if not (x in seen or seen.add(x))]

# Package metadata
setup(
    name="microscopy-anisotropy-3d",
    version="1.1.0",  # Updated version for OME-Zarr support
    author="Microscopy Analysis Team",
    author_email="analysis@microscopy.org",
    description="GPU-accelerated 3D microscopy anisotropy analysis with OME-Zarr + NGFF multiscale output",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/microscopy-anisotropy-3d",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/microscopy-anisotropy-3d/issues",
        "Documentation": "https://microscopy-anisotropy-3d.readthedocs.io/",
        "Source Code": "https://github.com/your-org/microscopy-anisotropy-3d",
        "OME-Zarr Spec": "https://ngff.openmicroscopy.org/latest/",
    },

    # Package discovery
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*']),
    py_modules=['functions_2', 'main_2'],

    # Requirements
    python_requires=">=3.8",
    install_requires=install_requires,

    # Optional dependencies
    extras_require={
        'dev': DEV_REQUIREMENTS,
        'docs': DOC_REQUIREMENTS,
        'cloud': CLOUD_REQUIREMENTS,
        'performance': PERFORMANCE_REQUIREMENTS,
        'all': DEV_REQUIREMENTS + DOC_REQUIREMENTS + CLOUD_REQUIREMENTS,
        'gpu-cuda': ['torch[cuda]>=2.0.0', 'cupy>=11.0.0'],  # For NVIDIA GPU support
        'minimal': [  # Minimal installation without napari and OME-Zarr
            'numpy>=1.21.0',
            'torch>=2.0.0',
            'tifffile>=2021.7.2',
            'tqdm>=4.62.0',
            'psutil>=5.8.0',
            'joblib>=1.1.0',
        ],
        'zarr-only': [  # Just OME-Zarr dependencies without full install
            'ome-zarr>=0.8.0',
            'zarr>=2.12.0',
            'dask[array]>=2022.8.0',
            'scikit-image>=0.19.0',
        ]
    },

    # Package data
    package_data={
        '': ['*.md', '*.txt', '*.yml', '*.yaml'],
    },
    include_package_data=True,

    # Entry points for command-line usage (optional)
    entry_points={
        'console_scripts': [
            'anisotropy-3d=main_2:main',
            'anisotropy-legacy=main:main',  # Keep legacy version available
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Framework :: napari",
    ],

    # Keywords for discovery
    keywords=[
        "microscopy", "anisotropy", "3d-analysis", "structure-tensor",
        "eigen-decomposition", "gpu-acceleration", "pytorch", "napari",
        "fractional-anisotropy", "medical-imaging", "computer-vision",
        "ome-zarr", "ngff", "multiscale", "cloud-storage", "zarr"
    ],

    # License
    license="MIT",

    # Zip safety
    zip_safe=False,

    # Additional metadata
    platforms=["any"],

    # Testing
    test_suite="tests",
    tests_require=[
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
    ],

    # Build requirements
    setup_requires=[
        'setuptools>=45.0.0',
        'wheel>=0.36.0',
    ],
)


# Post-installation checks and messages
def post_install_check():
    """Perform post-installation system checks."""
    print("\n" + "=" * 70)
    print("3D MICROSCOPY ANISOTROPY ANALYSIS v1.1.0 - INSTALLATION COMPLETE")
    print("Now with OME-Zarr + NGFF multiscale output support!")
    print("=" * 70)

    # Check PyTorch installation
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed successfully")

        # Check MPS availability on macOS
        if sys.platform == 'darwin':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("✓ Apple MPS backend available for GPU acceleration")
            else:
                print("⚠ Apple MPS backend not available, will use CPU")

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda} available")

    except ImportError:
        print("✗ PyTorch installation failed")

    # Check core dependencies
    core_deps = [
        ('numpy', 'NumPy'),
        ('tifffile', 'TIFF file support'),
        ('napari', 'Napari visualization'),
        ('joblib', 'CPU parallelization'),
    ]

    for module_name, description in core_deps:
        try:
            __import__(module_name)
            print(f"✓ {description} available")
        except ImportError:
            print(f"✗ {description} not available")

    # Check OME-Zarr dependencies
    zarr_deps = [
        ('ome_zarr', 'OME-Zarr support'),
        ('zarr', 'Zarr storage format'),
        ('dask', 'Dask distributed computing'),
        ('skimage', 'Scikit-image processing'),
    ]

    print("\nOME-Zarr + NGFF Dependencies:")
    for module_name, description in zarr_deps:
        try:
            __import__(module_name)
            print(f"✓ {description} available")
        except ImportError:
            print(f"✗ {description} not available")

    # Check optional cloud storage
    cloud_deps = [
        ('s3fs', 'AWS S3 storage'),
        ('gcsfs', 'Google Cloud Storage'),
        ('adlfs', 'Azure Data Lake Storage'),
    ]

    any_cloud = False
    for module_name, description in cloud_deps:
        try:
            __import__(module_name)
            print(f"✓ {description} available")
            any_cloud = True
        except ImportError:
            pass

    if any_cloud:
        print("✓ Cloud storage backends available")
    else:
        print("○ Cloud storage backends not installed (optional)")

    # System information
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count()
        print(f"\n✓ System: {cpu_count} CPU cores, {memory_gb:.1f} GB RAM")
    except ImportError:
        pass

    print("\nOutput Formats Supported:")
    print("✓ NumPy compressed (.npz) - Legacy format")
    print("✓ OME-Zarr (.ome.zarr) - NGFF multiscale format")
    print("✓ Interactive visualization - Napari viewer")

    print("\nTo get started:")
    print("1. Place your TIFF file in Data/your_file.tif")
    print("2. Update DATA_PATH in main_2.py")
    print("3. Run: python main_2.py")
    print("4. Or use: anisotropy-3d (command line)")

    print(f"\nDocumentation:")
    print("• Project: https://microscopy-anisotropy-3d.readthedocs.io/")
    print("• OME-Zarr: https://ngff.openmicroscopy.org/latest/")
    print("• NGFF Spec: https://github.com/ome/ngff")
    print("=" * 70)


# Run post-install check if this is being run directly
if __name__ == "__main__":
    post_install_check()
