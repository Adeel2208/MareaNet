from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="marea-net",
    version="5.5.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Marine-Aware REsilient Architecture for Underwater Semantic Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/marea-net",
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
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
        "opencv-python>=4.6.0",
        "Pillow>=9.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
)
