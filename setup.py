# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ccfv3-aligner",
    version="3.5.0",
    author="Axis Ju",
    author_email="axisju@example.com",
    description="High-Precision Interactive 3D DAPI Population Template & Deep Learning Brain Slice Registration Studio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AxisJu/CCFv3-aligner",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "pandas>=1.4.0",
        "scikit-image>=0.19.0",
        "matplotlib>=3.5.0",
        "torch>=1.12.0",
        "SimpleITK>=2.2.0",
        "tifffile>=2022.5.4",
        "imageio>=2.19.0",
        "tqdm>=4.64.0",
    ],
    entry_points={
        "console_scripts": [
            "ccfv3-studio=scripts.run_studio:main",
            "ccfv3-predict=scripts.predict_slice:main",
        ],
    },
)
