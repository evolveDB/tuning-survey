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


## Relevant Papers

### Knob Selection

* SARD: A statistical approach for ranking database tuning parameters (ICDEW, 2008)

* Too Many Knobs to Tune? Towards Faster Database Tuning by Pre-selecting Important Knobs (HotStorage 2020)

* Automatic Database Management System Tuning Through Large-scale Machine Learning (SIGMOD 2017)

### Feature Selection

* Automatic Database Management System Tuning Through Large-scale Machine Learning (SIGMOD 2017)

### Tuning Methods

* OpenTuner: An Extensible Framework for Program Autotuning (PACT, 2014)

* BestConfig: Tapping the Performance Potential of Systems via Automatic Configuration Tuning (SoCC, 2017)

* Tuning Database Conﬁguration Parameters with iTuned. (VLDB, 2009)

* An End-to-End Automatic Cloud Database Tuning System Using Deep Reinforcement Learning (SIGMOD 2019)

* QTune: A Query-Aware Database Tuning System with Deep Reinforcement Learning (VLDB 2019)

* Dynamic Configuration Tuning of Working Database Management Systems (2020)

* iBTune: Individualized Buffer Tuning for Large-scale Cloud Databases (VLDB 2019)

* Black or White? How to Develop an AutoTuner for Memory-based Analytics (SIGMOD 2020)

* The Case for NLP-Enhanced Database Tuning: Towards Tuning Tools that "Read the Manual" (VLDB 2021)

* DB-BERT: a Database Tuning Tool that “Reads the Manual” (SIGMOD 2022)

* CGPTuner: a Contextual Gaussian Process Bandit Approach for the Automatic Tuning of IT Configurations Under Varying Workload Conditions (VLDB 2021)

* Universal Database Optimization using Reinforcement Learning (VLDB 2021)

* ResTune: Resource Oriented Tuning Boosted by Meta-Learning for Cloud Databases (SIGMOD 2021)

* LlamaTune: Sample-Efficient DBMS Configuration Tuning (VLDB 2022)

### Tuning Transferring

* HUNTER- An Online Cloud Database Hybrid Tuning System for Personalized Requirements (2022 SIGMOD )

* ResTune: Resource Oriented Tuning Boosted by Meta-Learning for Cloud Databases (SIGMOD 2021)

* Towards Dynamic and Safe Configuration Tuning for Cloud Databases (SIGMOD 2022)


### Tutorial/Survey/Experiments

* An inquiry into machine learning-based automatic configuration tuning services on real-world database management systems (VLDB 2021)

* Facilitating Database Tuning with Hyper-Parameter Optimization- A Comprehensive Experimental Evaluation (2021 VLDB)

* Speedup Your Analytics: Automatic Parameter Tuning for Databases and Big Data Systems (tutorial)

* A Survey on Automatic Parameter Tuning for Big Data Processing Systems (survey for big-data-system tuning)


## Issues

Major improvement/enhancement in future.

* add more algorithms
* verify the correctness and test the algorithms

## Citation

```bash
@article{zhao2023automatic,
  title={Automatic Database Knob Tuning: A Survey},
  author={Zhao, Xinyang and Zhou, Xuanhe and Li, Guoliang},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```

## Contributors

We thank all the contributors to this project, more contributors are welcome!
