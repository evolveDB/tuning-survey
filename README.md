# tuning-survey

<p>
    <a href="#Overview">Overview</a>
    <a href="#Installation">Installation</a>
    <a href="#Datasets">Dataset</a>
    <a href="#Issues">Issues</a>
    <a href="#Citation">Citation</a>
</p>

![version](https://img.shields.io/badge/version-v1.0.0-blue)

## What's New?

* August 2022: First Commit

## Overview

In this survey, we implemented and tested existing methods in learned configuration tuning. The methods are divided into three categories, i.e., knob selection algorithms, feature selection algorithms, and tuning algorithms.


## Installation

```
pip install -r requirements.txt
```

## What can you do via this repository?

* Run the tests and read test codes in ./Tests folder
* Change config.ini so that the algorithms could run on your own machine. Also, you can choose the knobs that needs to tune and add some constraints on the knob value(e.g., min value, max value) if you want.

* Design new methods based on previous algorithms
* Compare the impact of different data preprocessing methods on knob selection methods(e.g., Standardization, Z-scores Standardization, min-max Standardization on Lasso) 

## Datasets

We have some datasets in the ./Workload folder. You can also add your own dataset when deploying on your own machine.

## Issues

Major improvement/enhancement in future.

* add more algorithms
* verify the correctness and test the algorithms

## Citation

## Contributors

We thank all the contributors to this project, more contributors are welcome!
