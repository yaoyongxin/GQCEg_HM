# GQCE(g)-Hubbard Model
Codes and job folders to reproduce the data in the following preprint:
* I.-C. Chen, A. Khindanov, C. Salazar, H. M. Barona, G. Harrab, F. Zhang, C.-Z. Wang, T. Iadecola, N. Lanatà, and Y.-X. Yao, Quantum-Classical Embedding via Ghost Gutzwiller Approximation for Enhanced Simulations of Correlated Electron Systems, [arXiv:2506.01204 (2025)](https://arxiv.org/abs/2506.01204)

## Credits to pre-exisiting codes
* Classical implementations of gGA method for single-band Hubbard model, [https://gitlab.com/collaborations3/g-ga-hubbard](https://gitlab.com/collaborations3/g-ga-hubbard)
* Adaptive Variational Quantum Imaginary-Time Evolution (AVQITE) code in CyQC: Quantum computing toolset for correlated materials simulations by Yao, Yong-Xin; De Andrade Getelina, Joao Carlos; Mukherjee, Anirban; Gomes, Niladri; Khindanov, Aleksei; Mootz, Martin; et al. (2024). figshare. Software. [https://doi.org/10.6084/m9.figshare.26298763](https://doi.org/10.6084/m9.figshare.26298763).

## Codes used in this work
* Folder **avqite**: contains the isolated AVQITE code and analysis script for density matrix.
* Folder **B3_ED_SV**: contains the adapted gGA code for gGA calculations with ghost orbital parameter *B=3*, quantum computing interface, and analysis scripts.

## Example for steps to reproduce the GQCE(g) calculaitons of Hubbard model
* To reproduce the gGA calculaiton of Hubbard model with *U=2.5* and *B=3*, move to folder **B3_ED_SV**, and execute `python ga_main.py`. 
* To generate the input file (incar) for AVQITE simulations, execute `python qc_solver.py`.
* To reproduce the data in file dm_vs_gates.dat: 
    * Execute `python ../avqite/run.py -c 0.01 -t 0.01 -n N -v 1e-5` consecutively, where `N` is to be replaced with the number of unitaries given in that data file. This concludes the AVQITE calculation and the solution is saved in ansatz.h5 and other equivalent files in different format for compatability reason.
    * Execute `python ../avqite/analysis.py` to post-analysis the density matrix and double occupancy. 
    * Execute `python analysis_rmat.py` to calculate the renormalization R-matrix. Note: the AVQITE solutions with increasing number of layers of gates are stored in folder **ans_cyc1_u2.5**. 

* To reproduce the noise model simulations, move to folder **B3_NM_QPU/L50**, execute `python dm_noise_model.py`. One needs to change the noise scaling factor *scale* in the script to produce the data given in files *results_L50_nm_{scale}*, which are used as input for subsequent spectral function analysis.
* To reproduce the noisy simulation results with Iceberg code for quantum error detection, move to folder **B3_NM_QPU**, and execute scripts `Ice_berg_exp_val_n_meas_{m}.py` with `m` replaced with the number of syndrome measurements. 
* For the density matrix measurement on IBM and Quantinuum emulators and H1-1, move to folder **B3_NM_QPU/L30**, and adapt the code `dm_IBM.py` (for IBM) and `dm.py` (for Quantinuum-nexus) with your credentials. The results are saved in *results_L30_{label}* files.