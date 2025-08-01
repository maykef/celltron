#!/usr/bin/env python3
"""
Setup script for 3D Microscopy Anisotropy Analysis
GPU-accelerated structure tensor computation with CPU parallelized eigen-decomposition
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
    return "3D Microscopy Anisotropy Analysis with GPU acceleration and CPU parallelization"


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
    'structure-tensor>=0.2.0',
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
    version="1.0.0",
    author="Microscopy Analysis Team",
    author_email="analysis@microscopy.org",
    description="GPU-accelerated 3D microscopy anisotropy analysis with CPU parallelized eigen-decomposition",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/microscopy-anisotropy-3d",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/microscopy-anisotropy-3d/issues",
        "Documentation": "https://microscopy-anisotropy-3d.readthedocs.io/",
        "Source Code": "https://github.com/your-org/microscopy-anisotropy-3d",
    },

    # Package discovery
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*']),
    py_modules=['functions', 'main'],

    # Requirements
    python_requires=">=3.8",
    install_requires=install_requires,

    # Optional dependencies
    extras_require={
        'dev': DEV_REQUIREMENTS,
        'docs': DOC_REQUIREMENTS,
        'all': DEV_REQUIREMENTS + DOC_REQUIREMENTS,
        'gpu-cuda': ['torch[cuda]>=2.0.0'],  # For CUDA support if needed
        'minimal': [  # Minimal installation without napari
            'numpy>=1.21.0',
            'torch>=2.0.0',
            'tifffile>=2021.7.2',
            'tqdm>=4.62.0',
            'psutil>=5.8.0',
            'structure-tensor>=0.2.0',
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
            'anisotropy-3d=main:main',
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
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
    ],

    # Keywords for discovery
    keywords=[
        "microscopy", "anisotropy", "3d-analysis", "structure-tensor",
        "eigen-decomposition", "gpu-acceleration", "pytorch", "napari",
        "fractional-anisotropy", "medical-imaging", "computer-vision"
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
    print("\n" + "=" * 60)
    print("3D MICROSCOPY ANISOTROPY ANALYSIS - INSTALLATION COMPLETE")
    print("=" * 60)

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

    # Check other critical dependencies
    critical_deps = [
        ('numpy', 'NumPy'),
        ('tifffile', 'TIFF file support'),
        ('napari', 'Napari visualization'),
        ('structure_tensor', 'Structure tensor library'),
    ]

    for module_name, description in critical_deps:
        try:
            __import__(module_name)
            print(f"✓ {description} available")
        except ImportError:
            print(f"✗ {description} not available")

    # System information
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count()
        print(f"✓ System: {cpu_count} CPU cores, {memory_gb:.1f} GB RAM")
    except ImportError:
        pass

    print("\nTo get started:")
    print("1. Place your TIFF file in Data/thresholded_image.tif")
    print("2. Run: python main.py")
    print("3. Or use: anisotropy-3d (if installed with pip)")

    print(f"\nFor documentation visit:")
    print("https://microscopy-anisotropy-3d.readthedocs.io/")
    print("=" * 60)


# Run post-install check if this is being run directly
if __name__ == "__main__":
    post_install_check()
