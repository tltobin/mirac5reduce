# mirac5reduce
Data Reduction for MIRAC-5 Instrument

Disclaimer: Very early stages, still under development!

## Required and Recommended Packages

The following python packages are required for for `mirac5reduce`:

-NumPy

-matplotlib

-astropy

The following packages are recommended but not required:

-Jupyter Notebook - a sample jupyter notebook is located in `docs` to provide easy access to the main functions. However, it isn't required.

-SAOImage DS9 - for viewing and examining fits files

-Aperture Photometry Tool (APT) - Interactive GUI for source and sky photometry

## Installation

To install mirac5reduce, download the package from GitHub, go into its top-level directory, and enter:
```
pip install -e .
```
The `-e` installs the package in *developer* mode, which will add the path to the downloaded package to your python interpreter, so that it knows where to import the package from.

## Getting Started

In the `docs` directory, you will find the `m5r_helper.ipynb` jupyter notebook, which will walk you through getting started.

The other document in the `docs` directory is an example configuration file, `runparams.init`, with explanations for the various parameters that can be set within it.
