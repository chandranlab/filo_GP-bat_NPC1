<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Sequence to feature vector</h3>
  
  <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/blob/main/img/features.png">
    <img src="/img/features.png" alt="Logo" width="400">
  </a>
  
  <p align="center">
    A collection of script to describe interfacial residues at the filovirus GP - bat NPC1 interface by their physicochemical properties
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

 Perl version: 5.30.2

 Python version: 3.7.6

 Python packages

|Package         | Version  |
|----------------|:--------:|
|bio             | 1.79     |
|mlxtend         | 0.19.0   |
|numpy           | 1.21.6   |
|pandas          | 1.0.1    |
|re              | 2.2.1    |
|seaborn         | 0.11.2   |


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ########################################################################################## -->

<!-- INPUT -->

## Input files

### PDB file (5F1B)
* Crystal structure of Zaire GP (chain A: GP1; chain B: GP2) bound to human NPC1 (chain C)

### Multiple Sequence alignments
* 1\_gp1.aln: MSA on filovirus GP sequences (CLUSTAL format)
* 1\_npc1.aln: MSA on bat NPC1s included in the experimental set (CLUSTAL format)
* 2\_auc.txt: Area-Under-the-Curve for the sigmoidal curve fitted in the binding ELISA experiments


### Other
* 0\_amino\_acid\_table.xlsx: Amino acid nomenclature conversion
* 0\_amino\_acid\_scales.xlsx: Amino acid physicochemical scales
* 0\_npc1\_list.xlsx: NPC1 nomenclature mapping (Bat specie <-> Internal ID)

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
