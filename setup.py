from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="marea-net",
    version="5.5.0",
    author="Adeel Mukhtar, Usman Ali",
    author_email="2021bme123@student.uet.edu.pk",
    description=(
        "Marine-Aware Resilient Architecture for Underwater Semantic Segmentation "
        "(CVPR 2026)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adeelmukhtar/marea-net",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.12.0,<2.16.0",
        "numpy>=1.23.0,<2.0.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.4.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
