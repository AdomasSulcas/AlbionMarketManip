#!/usr/bin/env python3
"""
Setup script for Albion Online Market Analysis package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Machine learning system for detecting market manipulation in Albion Online"

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "statsmodels>=0.13.0",
        "scipy>=1.9.0"
    ]

setup(
    name="albion-analyzer",
    version="1.0.0",
    description="Machine learning system for detecting market manipulation in Albion Online",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adomas Šulcas",
    author_email="",
    url="",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ]
    },
    entry_points={
        "console_scripts": [
            "albion-analyzer=main:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="albion online, market analysis, fraud detection, machine learning, gaming",
    project_urls={
        "Bug Reports": "",
        "Source": "",
    },
)