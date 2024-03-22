# Physics-Transfer Learning for Material Strength Screening

![PT](./GA.png)

## Overview

The strength of materials, like many problems in the natural sciences, spans multiple length and time scales, and the solution has to balance accuracy and performance.
Material screening by strength from first-principles calculations is computationally intractable for the nonlocal nature of strength, and not included in the state-of-the-art computational material databases.
To address this challenge, we propose a physics-transfer (PT) framework to learn the physics from empirical atomistic simulations and then predict the strength from chemically accurate density functional theory-based calculations of material parameters.
Notably, the strengths of single-crystalline metals can be predicted from a few single-point calculations for the deformed lattice and on the γ surface, allowing efficient, high-throughput screening for material discovery.
This physics-transfer framework can be generalized to other problems facing the accuracy-performance dilemma, by harnessing the hierarchy of physics in the multiscale models of materials science.

## Installation

First, download the repository from GitHub.
```
git clone https://github.com/zhaoyj21/PT.git
cd ./PT
```

The codes for PT learning were coded in `python 3.9`.
The `PT.yml` file is provided to create an environment with the required packages.
```
conda env create -f ./PT.yml
```

## Building digital libraries

To construct the digital libraries, a wide spectrum of metals with crystalline structures of fcc (Cu, Ni, Al, Au, Pd, Pt), bcc (Fe, Mo, Ta, W), and hcp (Ti, Mg, Zr, Co) is explored.
The elastic constants, $\gamma$ surfaces, and Peierls stresses are calculated using empirical force fields such as EAM and MEAM with parameters reported from [different sources](https://www.ctcms.nist.gov/potentials/) [1].

To calculate the $\gamma$ surfaces, you can run the code on the HPC (high-performance computing) cluster with SLURM (Simple Linux Utility for Resource Management).
```
cd ./PT/MD/gsf
sbatch sub.sh
```
To calculate the Peierls stresses, you can run the following code.
```
cd ./PT/MD/shear
sbatch sub.sh
```
Note: 
- The potential file `cu.eam` can be replaced for other potential parameters and metals from [different sources](https://www.ctcms.nist.gov/potentials/).
- The file `in.lmp` is the input script of LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) to calculate the $\gamma$ surfaces. For metals with different lattice structures, lattice constants, and mass, you should modify this template file.
- The file `sub.sh` is the script file that submits a task in the SLURM scheduling system, which specifies the number of computing resources the task occupies.

To extract the $\gamma$ surfaces, and Peierls stresses from the MD (molecular dynamics) results, you can find the MATLAB code in the following directory.
```
cd ./PT/Processing
```
Note: 
- The MATLAB script `process_gamma_surface.m` is used to extract $\gamma$ surfaces.
- The MATLAB script `detect_dislocation_move.m` is used to identify the movements of dislocations and extract Peierls stresses.
- For metals with different lattice structures and lattice constants, you should modify these template files.

Finally, the digital libraries of MD results are saved as the file `MD_libraries.mat`.

The neuroevolution-potential (NEP) framework is adopted to develop Machine Learning Force Fields (MLFFs) for fcc Cu, Al, bcc Fe, and hcp Ti [2].
The well-trained MLFFs are applied to calculate the $\gamma$ surfaces, and Peierls stresses utilizing the same scripts only change the EAM/MEAM potentials to MLFFs potentials.
The scripts and potentials files `nep.txt` can be found in the following directory.
```
cd ./PT/MLFFs
```
Finally, the digital libraries of MLFF results are saved as the file `MLFFs_libraries.mat`.

To validate the hypothesis and feasibility of the PT framework, we directly calculate the Peierls stresses in small systems (`S', with 244 atoms).
You can run the VASP (Vienna Ab initio Simulation Package) scripts in the following directory.
```
cd ./PT/DFT
sbatch sub.sh
```
Note: 
- The `INCAR`, `POSCAR`, `POTCAR`, `KPOINTS` are input, structure, pseudopotential, and mesh files, respectively.
- The `strength.py` controls step-wise strain loading and the `input.dat` defines the loading increment and number of steps.

## Training

You can train the PTNN with `main.py` in the following directory.
```
cd ./PT/PTNN
python main.py
```
Note: 
- The file `model_config.py` defines the hyperparameters including learning rate, batch size, etc.
- The file `model.py` defines the architectures of neural networks.
- The files `train_model.py` and `eval_model.py` describe the training and evaluation of models.
- The files `utils.py` and `AWD.py` are utilized to define tools and metrics to estimate errors according to the activity-weight duality [3].
- The file `FT.py` is utilized to fine-tune the physics from MD with MLFFs data.

After training, the parameters files of models can be found in the directory `./PT/PTNN/models`.

## Inference
 
You can transfer the physics and predict the target at chemical accuracy levels with the code in the following directory.
```
cd ./PT/PTNN
python pred-dft-correct.py
```
Note: For different models, you should change the descriptions of model files in the directory `./PT/PTNN/models`.

## References

- [1] Becker, C. A., Tavazza, F., Trautt, Z. T., & de Macedo, R. A. B. (2013). [Considerations for choosing and using force fields and interatomic potentials in materials science and engineering](https://www.sciencedirect.com/science/article/abs/pii/S1359028613000788), Current Opinion in Solid State and Materials Science, 17(6), 277-283.
- [2] Fan, Z., Zeng, Z., Zhang, C., Wang, Y., Song, K., Dong, H., ... & Ala-Nissila, T. (2021). [Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.104309), Physical Review B, 104(10), 104309.
- [3] Feng, Y., Zhang, W., & Tu, Y. (2023). [Activity–weight duality in feed-forward neural networks reveals two co-determinants for generalization](https://www.nature.com/articles/s42256-023-00700-x), Nature Machine Intelligence, 5(8), 908-918.
