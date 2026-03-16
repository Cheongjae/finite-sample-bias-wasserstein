# Supplementary Materials for AISTATS 2026

This repository contains supplementary materials for the AISTATS 2026 paper:

**On the Finite-Sample Bias of Minimizing Expected Wasserstein Loss Between Empirical Distributions**

## Requirements

The code was tested with **Python 3.8**.

Required packages:
- `numpy`
- `scipy`
- `mpmath`
- `torch`
- `pot`
- `geomloss`
- `pandas`
- `yfinance`
- `multitasking` (`<= 0.0.11`)
- `seaborn`

## Repository Guide

This repository contains the code and notebooks used to generate the figures in the paper and appendix.

---

## Main Paper

### Chapter 3: Gaussian, well-specified case

- **Figures 2, 3(a)**  
  `location_scale_pdfs_empirical_debias.ipynb`

- **Figures 3(b), 8(a), 8(b)**  
  `sgd_gaussian.ipynb`  
  `sgd_gaussian_debias.ipynb`

- **Figure 3(c)**  
  Computation:
  - `main_GMMfit.py`
  - `train_GMMfit.py`
  - `utils_GMMfit.py`  

  Plotting:
  - `plot_GMMfitting_results.ipynb`

Example command:
```bash
python3 main_GMMfit.py --K 4 --pi 0.4 0.3 0.2 0.1 --dim 2 --mb_size 32 --Nsteps 20000 --debias --use_init --seed 0
```

---

### Chapter 4: Tukey g-and-h, misspecified case

- **Figure 4**  
  `tukey_misspecified_realdata.ipynb`

---

### Chapter 4: Neural network generator, misspecified case

- **Figure 5**  
  Computation:
  - `main_toy.py`
  - `utils_GMMfit.py`

  Plotting:
  - `plot_toy_results.ipynb`

Example command:
```bash
python main_toy.py --datatype GMM --K 4 --pi 0.4 0.3 0.2 0.1 --dim 2 --z_dim 10 --mb_size 32 --Nsteps 10000 --dist Wasserstein --opt AdamW --stream_data --seed 0
```

---

### Chapter 5: Gaussian, Sinkhorn divergence, well-specified case

- **Figures 6(a), 6(b)**  
  Computation:
  - `compute_entreg_full_parallel.ipynb`

  Plotting:
  - `plot_entreg_full_values_gaussian.ipynb`

- **Figure 6(c)**  
  Computation:
  - `main_GMMfit.py`
  - `train_GMMfit.py`
  - `utils_GMMfit.py`

  Plotting:
  - `plot_GMMfitting_entreg_results.ipynb`

Example command:
```bash
python3 main_GMMfit.py --K 4 --pi 0.4 0.3 0.2 0.1 --dim 2 --mb_size 32 --Nsteps 50000 --debias --use_init --reg 1.0 --scaling 0.5 --seed 0
```

---

## Appendix

### Appendix B.4.1: Gaussian, misspecified case

- **Figure 7**  
  `location_scale_misspecified.ipynb`

- **Figures 8(c), 8(d)**  
  `sgd_gaussian_debias_misspecified.ipynb`

---

### Appendix B.4.2: Gaussian, semi-discrete case

- **Figures 9, 10**  
  `location_scale_semidiscrete.ipynb`

- **Figure 11**  
  `sgd_gaussian_debias_semidiscrete.ipynb`

---

### Appendix B.4.3: Gaussian, Sinkhorn divergence, misspecified case

- **Figure 12**  
  Computation:
  - `compute_entreg_full_misspecified_parallel.ipynb`

  Plotting:
  - `plot_entreg_full_values_gaussian_misspecified.ipynb`

---

### Appendix B.5.2: Tukey g-and-h, well-specified case

- **Figures 13, 14**  
  `tukey_wellspecified.ipynb`

---

### Appendix B.5.3: Tukey g-and-h, misspecified case

- **Figures 15, 16, 17**  
  `tukey_misspecified_realdata.ipynb`

---

### Appendix B.6.1: Affine, discrete-discrete case

- **Figure 18**  
  `affine_pdfs_empirical_debias.ipynb`

- **Figure 19**  
  `sgd_affine.ipynb`  
  `sgd_affine_debias.ipynb`

---

### Appendix B.6.2: Affine, semi-discrete case

- **Figure 20**  
  `affine_semidiscrete_objplot.ipynb`

- **Figure 21**  
  `affine_semidiscrete_a_multi.ipynb`

- **Figure 22**  
  `sgd_affine_debias_semidiscrete.ipynb`

---

### Appendix B.7: Neural network generator, Sinkhorn divergence, misspecified case

- **Figure 23**  
  Computation:
  - `main_toy.py`
  - `utils_GMMfit.py`

  Plotting:
  - `plot_toy_results.ipynb`

Example command:
```bash
python main_toy.py --datatype GMM --K 4 --pi 0.4 0.3 0.2 0.1 --dim 2 --z_dim 10 --mb_size 32 --Nsteps 10000 --dist Sinkhorn --opt AdamW --stream_data --seed 0
```

---

## Notes

- The scripts and notebooks are organized by figure number for ease of reproduction.
- Some figures require running computation scripts first and then separate plotting notebooks.
- Random seeds are specified in the example commands where applicable.
- The Gaussian mixture settings used in multiple experiments are shared across several scripts.

## Acknowledgment

Some implementation details were written with reference to the following public repositories:

- `https://github.com/kilianFatras/minibatch_Wasserstein`
- `https://github.com/kimiandj/slicedwass_abc`
