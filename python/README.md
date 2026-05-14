## **Description**
This project contains a fully customizable continuous wavelet transform implementation for python.
It presents a CWT-based analysis, using the **Wavelet Area Interpretation**, and its possible applications for python.
It introduces a wavelet-based covariance reconstruction, based on the CWT reconstruction identity, adapted for computing eddy-covariance fluxes.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/jonathanbitton/Wavelets-WAI/python.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
Run the example notebook (notebooks/Example_WAI.ipynb or notebooks/Example_cov.ipynb), and make changes according to input data. 

## Example Code
See notebooks/Example_WAI.ipynb and notebooks/Example_cov.ipynb for detailed case-studies.

## File Structure
```
python/
├── README.md                  # Project description and usage instructions
├── requirements.txt           # List of dependencies
├── setup.py                   # For packaging and installation
├── .gitignore                 # Files and folders to ignore in Git
├── data/                      # Folder for example datasets
│   ├── BE_Lon                 # Folder containing the BE-Lon dataset (for covariance reconstruction)
│   ├── GPPdata.txt            # Example data file (for WAI)
│   ├── Hesse_Aux_Data.txt     # Driving variables at Hesse
│   └── Hesse_Flux_Data.txt    # Fluxes values at Hesse
├── notebooks/                 # Folder for Jupyter notebooks
│   ├── Example_WAI.ipynb      # Example notebook for WAI
│   └── Example_cov.ipynb      # Example notebook for covariance reconstruction
└── src/                       # Source code folder
    ├── __init__.py            # Package initialization file
    ├── cwtransform.py         # Main module for wavelet transform
    ├── wt_utils.py            # Utility functions for wavelets
    ├── data_utils.py          # Utility functions for data handling
    └── wai_utils.py           # Utility functions for wavelet area interpretation
```

## License
This project is licensed under the Apache 2.0 License.
