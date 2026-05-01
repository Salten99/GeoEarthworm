from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="geoearthworm",
    version="0.1.0",
    author="Samuel Alten",
    author_email="samuel.alten@gmail.com",
    description="Bedrock depth interpolation from scattered borehole measurements using IDW",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/geo-earthworm",
    py_modules=["geo_earthworm"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "rasterio>=1.3.0",
        "pandas>=1.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Intended Audience :: Science/Research",
    ],
    keywords="bedrock interpolation IDW geospatial GeoTIFF raster borehole geology",
)
