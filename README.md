<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/img/0_overview.png">
    <img src="/img/0_overview.png" alt="Logo" width="400">
  </a>

  <h3 align="center">Decoding the blueprint of filovirus entry through large-scale binding assays and machine learning</h3>

  <p align="center">
    A collection of scripts implemented to characterize and model the binding strength between filovirus glycoproteins and bat receptors (NPC1)
    <br />
    <a href="https://github.com/chandranlab/filo_GP-bat_NPC1"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/issues">Report Bug</a>
    ·
    <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/issues">Request Feature</a>
  </p>
</div>

<!-- ########################################################################################## -->

<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About this project</a>
    </li>
    <li>
      <a href="#scripts">Scripts</a>
      <ul>
        <li><a href="#process-elisa-readouts">Process ELISA readouts</a></li>
        <li><a href="#prepare-input-sequences-for-machine-learning">Prepare input sequences for machine learning</a></li>
        <li><a href="#random-forest">Random forest</a></li>
      </ul>
    </li>
    <li><a href="#datasets">Datasets</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

<!-- ########################################################################################## -->

<!-- ABOUT THE PROJECT -->
## About this project

Code repository for manuscript “Decoding the blueprint of filovirus entry through large-scale binding assays and machine learning” (Lasso et al., manuscript in preparation). This collection includes:

Evidence suggests that bats are important hosts of filoviruses, but the specific bats involved remain largely unknown. The Niemann-Pick C1 protein (NPC1) serves as an essential entry receptor, with amino acid variations at the virus-receptor interface influencing viral susceptibility and species-specific tropism. We reasoned that variation in virus-receptor binding would aid in identifying potential host species. We performed binding studies across seven filovirus glycoproteins (GPs) and NPC1 orthologs from 81 bat species, finding that GP-NPC1 binding agreed poorly with overall sequence identity or phylogeny. However, integrating binding assays with machine learning revealed genetic factors influencing virus-receptor binding. The implemented model predicts GP-NPC1 binding avidity (R² of 0.69-0.75). Experimental and predicted binding avidities, combined with data on bat distribution and past Ebola virus outbreaks, helped rank bat species' potential as Ebola virus hosts. This study provides a multidisciplinary approach for predicting susceptible species and guiding filovirus host surveillance.

<li>
Processing of experimental data
  <ul>
    <li>Processing of ELISA binding experiments</li>
    <li>Hierarchical clustering based on binding profiles</li>
  </ul>
Machine Learning
  <ul>
    <li>Training & Evaluation a Random Forest regressor to predict binding avidity between 
    filovirus GPs and bat NPC1-C</li>
  </ul>
</li>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ########################################################################################## -->

<!-- ABOUT THE PROJECT -->
## Scripts
#### Process ELISA readouts
<p>
<ul>
<li>A python script to automatically process enzyme-linked Immunosorbent assay (ELISA) output generated with <a href="https://explore.agilent.com/imaging-microscopy" target="_blank">Agilent Biotek’s cytation</a>. Processing includes denoising, normalization, and sigmoidal curve fit. <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/tree/main/scr/0_elisa">More info</a></li>
</ul>
</p>

#### Prepare input sequences for machine learning
<p>
<ul>
<li>A collection of Jupyter notebooks to translate amino acid sequences from filovirus GPs and host NPC1s into feature vectors that describe various amino acid physicochemical properies. Paired vectors will be used for training and evaluating a Random Forest. <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/tree/main/scr/1_seq_to_feat">More info</a></li>
</ul>
</p>

#### Random forest
<p>
description
</p>

#### Model interpretation
<p>
description
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ########################################################################################## -->

<!-- DATASETS -->
## Datasets
<div>
<p>
description
</p>
<p align="right">(<a href="#readme-top">back to top</a>)</p>
</div>

<!-- ########################################################################################## -->

<!-- CONTACT -->
## Contact

* Gorka Lasso - [@gorkalasso](https://twitter.com/gorkalasso) - gorka.lasso@gmail.com
* Kartik Chandran - [@chandranlab](https://twitter.com/chandranlab) - kartik.chandran@gmail.com

Project Link: [https://github.com/chandranlab/filo_GP-bat_NPC1](https://github.com/chandranlab/filo_GP-bat_NPC1)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ########################################################################################## -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/chandranlab/filo_GP-bat_NPC1/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/chandranlab/filo_GP-bat_NPC1.svg?style=for-the-badge
[forks-url]: https://github.com/chandranlab/filo_GP-bat_NPC1/network/members
[stars-shield]: https://img.shields.io/github/stars/chandranlab/filo_GP-bat_NPC1.svg?style=for-the-badge
[stars-url]: https://github.com/chandranlab/filo_GP-bat_NPC1/stargazers
[issues-shield]: https://img.shields.io/github/issues/chandranlab/filo_GP-bat_NPC1.svg?style=for-the-badge
[issues-url]: https://github.com/chandranlab/filo_GP-bat_NPC1/issues
