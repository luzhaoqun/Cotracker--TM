from setuptools import setup, find_packages

setup(
    name="tympanic_detection",
    version="0.1.0",
    description="Tympanic membrane deformation detection using CoTracker",
    packages=find_packages(include=["tympanic_detection", "tympanic_detection.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "matplotlib",
        "torch",
        "torchvision",
    ],
)
