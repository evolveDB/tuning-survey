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

In this survey, we implemented and tested existing methods in learned configuration tuning. The methods are divided into four categories, i.e., data pre-processing, knob selection algorithms, feature selection algorithms, and tuning algorithms.

<table>
    <tbody>
    <tr valign="top">
        <td>Pre-Processing</td>    
        <td>Knob Selection</td>
        <td>Feature Selection</td>
        <td>Tuning Algorithms</td>
    </tr>
    <tr valign="top">
        <td> Standardization <br> Z-scores Standardization <br> min-max Standardization </td>
        <td> Lasso <br> CART <br> P&B </td>
        <td> K-Means <br> QueryEmbed <br> R/W-Ratio <br> TF_IDF </td>
        <td> Heuristic <br> GaussianProcess <br> DNN <br> DDPG <br> Attention+DDPG </td>
    </tr>
    </tbody>
</table>


## Installation

```
pip install -r requirements.txt
```

## What can you do via this repository?

* Read and run the test codes in ./Tests folder.

* Change the config.ini file so that the algorithms could run on your own machine. Also, you can choose the knobs that need to tune and add some constraints on the knob value(e.g., min value, max value) if you want.

* Design new algorithms based on the previous tuning methods.

* Compare the impact of different data preprocessing algorithms in knob selection(e.g., Standardization, Z-scores Standardization, min-max Standardization on Lasso) 

## Datasets

We have some datasets in the ./Workload folder. You can also add your own dataset when deploying on your own machine.

## Issues

Major improvement/enhancement in future.

* add more algorithms
* verify the correctness and test the algorithms

## Citation

## Contributors

We thank all the contributors to this project, more contributors are welcome!
