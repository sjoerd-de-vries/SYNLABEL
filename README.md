# Synthetic Data for Label Noise Research

This repository contains the code for the SYNLABEL package.

As presented in the paper: Generating the Ground Truth: Synthetic Data for Label Noise Research

## Installation

The virtual environment that was used to run this framework successfully is specified in requirements.txt

The code was tested using Python version 3.9

To create a new virtual environment, use:

`$ python -m venv .venv`

Then activate the environment using:

`$ source .venv/bin/activate`

Then install the needed packages using:

`$ pip install -r requirements.txt`

The SYNLABEL package will automatically be installed as a locally editable package, along
with some additional packages required to run the experiments.

## Repository Structure

### **/data**

Contains the keel_vehicle dataset used in the experiments shown in Section 5 & 6 of the paper.

The user can freely add any dataset here and apply the framework to it instead.

### **/experiments**

Contains the code used to run the experiments presented in the paper.

***/analysis**

Contains the files used to analyse the results from the experiments in */demonstrations, reads these from */json, writes the results to */figures.

***/demonstrations**

Contains the files needed to conduct experiments with the framework. Results are saved in */json.

***/figures**

Contains the figures that are produces by the files in */analysis.

***/json**

Contains the results of the experiments in */demonstrations in .json format.

### **/src/synlabel**

The framework. 

***/dataset_types**

Contains the datasets that are defined in the framework.

***/transformations**

Contains the transformations that are allowed between datasets in the framework.

***/utils**

Contains helper methods.

## How to use

The files in "/experiments/demonstrations" are used to conduct the experiments for which the results are shown in Section 5 and 6 of the paper. 

The file ground_truth_generation.py shows how a simulated dataset can be generated and set as the Ground Truth, and how a learned function can be used to generate a Ground Truth dataset.

The files exp_feature_hiding.py and exp_injection_methods were used to generate the results for the paper, and illuatrate how a PG set can be constructed from a G set via feature_hiding.

When these files are executed, new result folders are created based on the current time. In order to analyse these, the paths in the files in */analysis will need to be adjusted.

## Additional Instructions

The framework contains some checks, but will be extended in the future.

The user is responsible for imputing any missing values in their dataset as not all methods may be able to handle those. Furthermore, the framework is suited to tabular, classification type problems.

It is recommended to recode the different classes into integer format in the range 0 to n_classes - 1.