## **Description**
This project presents a continuous wavelet transform implementation and a signal analysis method based on the **Wavelet Area Interpretation** for matlab.

## Installation
Clone the repository:
```bash
git clone https://github.com/jonathanbitton/Wavelets-WAI/matlab.git
cd matlab
```

## Usage
Run the example script (scripts/Example_Script.m), and make changes according to input data. 

## Example Code
See scripts/Example_Script.m for an application or notebooks/Example.ipynb from the python folder for a detailed case-study.

## File Structure
```
matlab/
├── scripts/                                         # Scripts folder
│   └── Example_Script.m                             # Example script
├── src/                                             # Main functions
│   ├── coefvalues.m                                 # Compute mean period bands
│   ├── ComputeWave.m                                # Compute wavelet in time domain
│   ├── cwtransform.m                                # main CWT functions
│   ├── DatesFromCenterToExtremumOrZeroDOG.m         # Date translation
│   ├── DefFourierF.m                                # Define Fourier Factor
│   ├── DiracCOIComputation.m                        # Compute Cone of Influence (COI) using dirac impulses
│   ├── DOGCOI.m                                     # Compute COI for DOG wavelets
│   ├── ExtractPeaks.m                               # Extract peaks in period bands
│   ├── GetArgs.m                                    # Extract and format user-defined arguments
│   ├── LaunchWAI.m                                  # Launch WAI-related computations and plot
│   ├── plotscalogram.m                              # Plot scalogram
│   ├── plotWAI.m                                    # Plot computations steps for wavelet coefficient to conduct WAI
│   ├── significance.m                               # Compute statistical significance (red background noise, according to Torrence & Compo, 1998)
│   ├── SupportWav.m                                 # Calculate wavelet support
│   └──UnitsAndFctHandles.m                          # Convert units and define function handles
├── data/                                            # Data folder
│   └── GPPdata.txt
├── setup.m                                          # Setup file for scritps
└── README.md                                        # Project documentation
```

## License
This project is licensed under the Apache 2.0 License.
