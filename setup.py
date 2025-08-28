
from setuptools import Command, find_packages, setup


setup(
    name="spatialFuser",
    version="1.0.0",
    descripton="SpatialFuster: is a unified framework for spatial multi-omics integrative analysis",
    url="https://github.com/RandyCai0408/SpatialFuser_doc",
    author="Wenhao Cai",
    author_email = "randy_caii@outlook.com",
    license="MIT",
    packages=['spatialFuser'],
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.8",
)