# PAC-Bayesian Binary Activated Deep Neural Networks

This repository provides an **implementation** of the core ideas from the paper:

> **Dichotomize and Generalize: PAC-Bayesian Binary Activated Deep Neural Networks**  
> *Gaël Letarte, Pascal Germain, Benjamin Guedj, and François Laviolette (NeurIPS 2019)*  
> [[arXiv:1905.10259]](https://arxiv.org/abs/1905.10259)

It serves as a **theoretical and practical study** on applying **PAC-Bayesian learning** to **Binary Activated Deep Neural Networks (BAMs)**. The central focus is on how to:
- Formulate **PAC-Bayesian** generalization bounds in the presence of **discrete sign activations**.
- Address **non-differentiability** by substituting the sign function with **smooth surrogates** (e.g., the Gaussian error function, *erf*).
- Use **Monte Carlo (MC) methods** to efficiently approximate and optimize over the **posterior** distribution of network parameters.

This project was developed as part of the **"Principes Théoriques de l’Apprentissage Profond"** course at CentraleSupélec (2024/25). It **summarizes**, **implements**, and **empirically evaluates** the key elements of the original paper, offering reproducible experiments and theoretical discussions.

---

## 📌 Project Overview

### 1. Theoretical Foundations

1. **PAC-Bayesian Framework**  
   - Provides *distribution-dependent* generalization guarantees by bounding the expected risk of an aggregated classifier.  
   - Involves minimizing a bound that trades off **empirical error** and a **KL divergence** (model complexity) term.

2. **Binary Activated Multilayer (BAM) Networks**  
   - Employ **binary (sign) activations**, which are notoriously **non-differentiable**.  
   - The **sign** function is replaced by a **stochastic `erf` surrogate**, enabling **gradient-based training**.

3. **Monte Carlo Approximation**  
   - Approximates the **intractable summations** across binary activations by sampling from a **Bernoulli** or related distribution.  
   - Avoids enumerating all possible hidden-layer configurations.

### 2. Objectives
- **Revisit the main results** of Letarte et al. (2019) in a more accessible code framework.  
- **Demonstrate** how the **Catoni-style PAC-Bayes bound** can be *effectively minimized* in practice.  
- **Compare** smooth (e.g., `tanh`) vs. binary (`sign`) activation performance, contrasting standard ERM with PAC-Bayesian training.  

---

## 🏗 Implementation

The code is primarily provided in the Jupyter notebook [`assignement_code.ipynb`](./assignement_code.ipynb). Key components include:

1. **Model Definition**  
   - `PBGNet` (PAC-Bayesian Gradient Network): A neural network that uses a differentiable *erf* approximation of binary activations and **explicitly minimizes** the PAC-Bayesian bound.  
   - `BaselineNet`: A standard feed-forward architecture (e.g., `tanh` activation) for comparison.

2. **Bounds and Losses**  
   - **Catoni’s Bound**: Implemented to provide a *non-vacuous* generalization measure, balancing empirical error and KL divergence to a prior.  
   - **Monte Carlo Estimation**: Used to approximate the forward pass and compute gradients for the `erf`-based sign surrogates.

3. **Additional Utilities**  
   - **DatasetLoader**: Loads and processes several benchmark datasets (MNIST variants, Adult, etc.).  
   - **Train Function**: Handles training loops, logging (via [Poutyne](https://poutyne.org/)), validation, checkpointing, and final testing.

---

## ⚙ **Setup and Dependencies**
To run the experiments locally, install the required dependencies:

```bash
pip install torch matplotlib seaborn numpy poutyne
```

> **Note:** This project uses **Poutyne**, a lightweight framework built on PyTorch to simplify training workflows.

---

## 🏗 **Usage**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Masrour-Aymane/PAC-Bayesian-Binary-Activated-Deep-Neural-Networks.git
   cd PAC-Bayesian-Binary-Activated-Deep-Neural-Networks
   ```

2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook assignement_code.ipynb
   ```

---

## 📊 Key Experiments in the Notebook

1. **Hyperparameter Selection & Final Model Configurations**  
   - The notebook begins by **loading a pre-determined `param_grid`** that contains only **the best hyperparameters** for each model and dataset combination.  
   - These configurations stem from a broader search (commented out in code), where:
     - **PBGNet** and **PBGNet-Pretrain** are chosen based on minimizing the **PAC-Bayes bound**.  
     - **PBGNet-ll** and **Baseline** are chosen based on minimizing the **validation loss**.
   - **No further tuning** is performed during the final runs; instead, these hyperparameters are used **as-is** to generate reproducible results.

2. **Bound vs. Empirical Loss Evaluation**  
   - The final analyses **compare the PAC-Bayes bound** to the **observed test loss** across datasets.
  
3. **Comparison of Activation Functions**  
  - Erf-based PAC-Bayesian aggregation
  - Tanh activation
  - Sign activation
    
4. **Empirical Study of Generalization Bounds**  


---

## 📜 References

1. **Letarte, G., Germain, P., Guedj, B., & Laviolette, F. (2019).**  
   *Dichotomize and Generalize: PAC-Bayesian Binary Activated Deep Neural Networks.* NeurIPS.  
   [[arXiv:1905.10259]](https://arxiv.org/abs/1905.10259)

2. **Catoni, O. (2007).**  
   *PAC-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning.* Institute of Mathematical Statistics.

3. **Germain, P., Bach, F. R., Lacasse, A., & Laviolette, F. (2009).**  
   *PAC-Bayesian Learning of Linear Classifiers.* In ICML.

---

## 🤝 Contributions & License

- This repository was developed as a **course assignment** at CentraleSupélec for the *"Principes Théoriques de l’Apprentissage Profond"* (2024/25) curriculum.  
- Feel free to open issues or submit pull requests for improvements.
- This project is released under the [MIT License](LICENSE). Please review `LICENSE` for usage and distribution details.

**If you find this work helpful, please consider giving a ⭐ star!**  
Happy researching and experimenting with **PAC-Bayesian Binary Activated Deep Neural Networks**.
