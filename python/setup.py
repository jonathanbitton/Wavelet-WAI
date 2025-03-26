from setuptools import setup, find_packages

setup(
    name="python",
    version="1.0.0",
    description="A Python package for continuous wavelet transform (CWT) and Wavelet Area Interpretation (WAI) analysis.",
    author="Jonathan Bitton",
    author_email="bitton.jonathan@outlook.com",
    url="https://github.com/jonathanbitton/Wavelet-WAI/python",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pyfftw>=0.12.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)