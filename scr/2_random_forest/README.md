<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Random Forest regressor</h3>
  
  <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/img/curves.png">
    <img src="/img/curves.png" alt="Logo" width="400">
  </a>
  
  <p align="center">
    A python code for the automatic processing of ELISA experiments
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

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

 Python version: 3.7.6

 Python packages

|Package         | Version  |
|----------------|:--------:|
|sklearn         | 1.0.2    |
|matplotlib      | 3.1.3    |
|numpy           | 1.21.6   |
|pandas          | 1.0.1    |
|seaborn         | 0.11.2   |


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ########################################################################################## -->

<!-- INPUT -->

## Input files

### model[1,2]_params.txt
* Optimized hyper-parameters for the two RF regressors presented in the manuscript (model 1 and model 2)

### model[1,2]_datasets
* model[1,2]\_train: Training dataset with selected features
* model[1,2]\_test: Testing dataset with selected features
* model[1,2]\_llov: Lloviu GP-binding datataset with selected features



<!-- ########################################################################################## -->

<!-- STEPS -->

## Steps

1. Identify, for each residue in chains A (GP1) and C (NPC1) the closest distance to binding partner
    1. Run perl script on PDB file
   ```sh
   ./1_pdb_resContacts_byDistance.pl -i input/5f1b.pdb -c1 A -c2 C -d 1000 -dif1 0 -dif2 372 > output/1_5f1b_interface_summary.txt
   ```
    2. Save output on excel spreadsheet (for exact details on formatting see output/1\_5f1b\_interface_summary)

2. Reduce amino acid sequences to interfacial residues (at a given cutoff)
    * Jupyter notebook: 2\_interface\_2\_fasta.ipynb

3. Add internal ID for bat NPC1s
    * Jupyter notebook: 3\_switch\_npc1header.ipynb

4. Pair interfacial fasta files (GP + NPC1)
    * Jupyter notebook: 4\_combine_fasta.ipynb

5. Convert paired fasta files into physicochemical properties (one at a time)
    * Jupyter notebook: 5\_seq\_2\_feature.ipynb

6. Add variable to predict (Binding AUC) to each feature assembled in the previous step
    * Jupyter notebook: 6\_add\_var\_2\_predict.ipynb

<p align="right">(<a href="#readme-top">back to top</a>)</p>
