from setuptools import setup, find_packages

setup(
    name="ditfastattn_api",
    version="0.1.0",
    author="hahnyuan",
    author_email="hahnyuan@gmail.com",
    description="DiTFastAttn API",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "diffusers",
        # Add other dependencies as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
