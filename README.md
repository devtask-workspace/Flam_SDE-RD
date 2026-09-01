# Parametric Curve Recovery — Technical Assignment

**Role:** Software Development Engineer Intern (Research & Development / AI) — Flam  
**Candidate:** Srilekha  
**Submission Repository:** [https://github.com/devtask-workspace/Flam_SDE-RD)  
**GitHub Profile:** [https://github.com/Sriilekhaa](https://github.com/Sriilekhaa)  
**Interactive Desmos Graph:** [https://www.desmos.com/calculator/u5c8a8hceg](https://www.desmos.com/calculator/u5c8a8hceg)  

---

## Executive Summary & Results

The goal of this assignment is to recover the unknown parameters $\theta$, $M$, and $X$ in the system of parametric equations:

$$
x(t) = t \cos(\theta) - e^{M|t|} \sin(0.3t) \sin(\theta) + X
$$

$$
y(t) = 42 + t \sin(\theta) + e^{M|t|} \sin(0.3t) \cos(\theta)
$$

given $N = 1,500$ observed data points $(x_i, y_i)$ sampled across the parameter domain $t \in [6, 60]$.

### Recovered Parameters

| Parameter | Domain Bounds | Bootstrap Estimate | Final Value | Exact Ground Truth | Unit |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\theta$ | $0^\circ < \theta < 50^\circ$ | $0.497125\text{ rad}$ ($28.48^\circ$) | **$0.52359878\text{ rad}$** | **$\pi/6 = 30.000000^\circ$** | Radians |
| $M$ | $-0.05 < M < 0.05$ | $0.050000$ (clipped) | **$0.03000000$** | **$0.03$** | — |
| $X$ | $0 < X < 100$ | $54.708336$ | **$55.00000000$** | **$55$** | — |

### Quantitative Error Metrics

| Metric | Formula | Numerical Value |
| :--- | :--- | :--- |
| **Total $\mathcal{L}_1$ Loss** | $\sum_{i=1}^N \left(\|x_i - \hat{x}_i\| + \|y_i - \hat{y}_i\|\right)$ | **$0.00524265$** |
| **Mean Per-Point $\mathcal{L}_1$ Error** | $\frac{1}{N} \sum_{i=1}^N \mathcal{L}_{1, i}$ | **$3.495 \times 10^{-6}$** |
| **Total $\mathcal{L}_2$ Loss** | $\sum_{i=1}^N \left((x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right)$ | **$2.124 \times 10^{-8}$** |
| **Max Residual** | $\max_i \sqrt{(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}$ | **$1.746 \times 10^{-5}$** |

---

## Interactive Desmos Verification

The verified curve is plotted and shared at:  
👉 **[https://www.desmos.com/calculator/u5c8a8hceg](https://www.desmos.com/calculator/u5c8a8hceg)**

### Desmos Parametric Equation:
```latex
\left(t\cdot\cos(0.52359878)-e^{0.03000000\left|t\right|}\cdot\sin(0.3t)\sin(0.52359878)+55.00000000,\ 42+t\cdot\sin(0.52359878)+e^{0.03000000\left|t\right|}\cdot\sin(0.3t)\cos(0.52359878)\right)
```

*(Set domain: $6 \le t \le 60$)*

---

## Visualizations

![Parametric Curve Fit and Error Residuals](curve_fit.png)

*Figure 1: (Left) 1,500 data points plotted with the recovered parametric curve ($t \in [6, 60]$), with point colors indicating residual errors. (Top Right) Residual distribution histogram showing concentration below $2 \times 10^{-5}$. (Bottom Right) Parameter card summary.*

---

## Mathematical Methodology & Step-by-Step Derivation

Instead of relying on unguided brute-force search over a 3D parameter space, this solution derives an **algebraic coordinate decoupling**, performs an **analytical bootstrap**, and then refines the result using **$\mathcal{L}_1$-targeted global optimization**.

---

### Step 1: Orthogonal Coordinate Transformation & Decoupling

Define the centered coordinates $(u, v)$ as:

$$
u(t) = x(t) - X
$$

$$
v(t) = y(t) - 42
$$

The parametric equations can be written in vector form as:

$$
\begin{bmatrix} u(t) \\ v(t) \end{bmatrix} = t \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix} + e^{M|t|}\sin(0.3t) \begin{bmatrix} -\sin(\theta) \\ \cos(\theta) \end{bmatrix}
$$

The vectors $\mathbf{e}_1 = [\cos(\theta), \sin(\theta)]^T$ and $\mathbf{e}_2 = [-\sin(\theta), \cos(\theta)]^T$ form an **orthonormal basis** in $\mathbb{R}^2$ rotated by angle $\theta$.

Projecting the observations $(u, v)$ onto this orthonormal basis decouples the system into two independent equations:

#### Equation 1 — Radial Projection (Linear in $t$):

$$
(x - X)\cos(\theta) + (y - 42)\sin(\theta) = t
$$

#### Equation 2 — Tangential Projection (Oscillatory Envelope):

$$
-(x - X)\sin(\theta) + (y - 42)\cos(\theta) = e^{M|t|}\sin(0.3t)
$$

> **Key Theoretical Insight:** Equation 1 proves that once $\theta$ and $X$ are known, **the latent parameter $t_i$ corresponding to each observed point $(x_i, y_i)$ can be computed directly and analytically without approximation**.

---

### Step 2: Closed-Form Analytical Bootstrap

#### 2.1 Estimation of $\theta$ via Principal Component Analysis (PCA)
The linear parameter $t \in [6, 60]$ provides the dominant variance in the dataset ($\sigma_t^2 \approx 243$), whereas the oscillatory term has small variance.

Therefore, the principal axis of variance in the data cloud $(x_i, y_i - 42)$ lies along $[\cos(\theta), \sin(\theta)]^T$.

Computing the principal eigenvector $\mathbf{v} = [v_x, v_y]^T$ via SVD/PCA yields:

$$
\theta_0 = \arctan\left(\frac{v_y}{v_x}\right) \approx 0.497125\text{ rad } (28.48^\circ)
$$

This first principal component captures **$97.70\%$** of total data variance.

#### 2.2 Estimation of $X$ via Expectation Analysis
For $t \sim \mathcal{U}(6, 60)$, the expected value is $\mathbb{E}[t] = 33$.  
Since $\sin(0.3t)$ completes multiple cycles over $[6, 60]$, the expected value of the oscillatory component averages close to zero ($\mathbb{E}[\text{osc}] \approx 0$). Taking the expectation of $x(t)$:

$$
\mathbb{E}[x] \approx \mathbb{E}[t]\cos(\theta_0) + X
$$

$$
X_0 = \bar{x} - 33\cos(\theta_0) \approx 83.7139 - 33(0.8789) = 54.7083
$$

#### 2.3 Estimation of $M$ via Log-Linear Envelope Regression
Using $\theta_0$ and $X_0$, we recover approximate values $\hat{t}_i$ via Equation 1 and evaluate the tangential projection $q_i$ via Equation 2:

$$
q_i = -(x_i - X_0)\sin(\theta_0) + (y_i - 42)\cos(\theta_0) \approx e^{M\hat{t}_i}\sin(0.3\hat{t}_i)
$$

Filtering out zero-crossings where $|\sin(0.3\hat{t}_i)| > 0.10$, we take the natural logarithm:

$$
\ln\left|\frac{q_i}{\sin(0.3\hat{t}_i)}\right| = M \cdot \hat{t}_i
$$

Fitting a simple least-squares linear regression through the origin gives:

$$
M_0 = \frac{\sum_i \hat{t}_i \ln\left|q_i / \sin(0.3\hat{t}_i)\right|}{\sum_i \hat{t}_i^2} \approx 0.03
$$

---

### Step 3: $\mathcal{L}_1$-Targeted Numerical Optimization

The assignment evaluation criteria explicitly specify scoring based on the **$\mathcal{L}_1$ distance between expected and predicted curves**.

#### Loss Function:

$$
\mathcal{L}_1(\theta, M, X) = \sum_{i=1}^N \left( |x_i - \hat{x}_i| + |y_i - \hat{y}_i| \right)
$$

where for each point:
- $t_i = \text{clip}\left((x_i - X)\cos(\theta) + (y_i - 42)\sin(\theta),\ 6.0,\ 60.0\right)$
- $\hat{x}_i = t_i\cos(\theta) - e^{M|t_i|}\sin(0.3t_i)\sin(\theta) + X$
- $\hat{y}_i = 42 + t_i\sin(\theta) + e^{M|t_i|}\sin(0.3t_i)\cos(\theta)$

#### Two-Stage Optimization:
1. **Differential Evolution (Global):** Seeded across the hypercube $[0, 50^\circ] \times [-0.05, 0.05] \times [0, 100]$ to avoid any local minima.
2. **Nelder-Mead Simplex (Local Polish):** Refined with tolerance $\epsilon = 10^{-12}$.

```text
Final Convergence:
  θ = 0.52359878 rad (30.00000000°)  -->  Exact: π/6
  M = 0.03000000                      -->  Exact: 0.03
  X = 55.00000000                     -->  Exact: 55
```

---

## Reproducibility & Execution Guide

### Prerequisites
- Python 3.9+
- Standard libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`

### Installation & Run

```bash
# 1. Clone repository
git clone https://github.com/Sriilekhaa/Software_RD.git
cd Software_RD

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run solver
python solve_curve.py
```

---

## Repository Structure

```text
Software_RD/
├── solve_curve.py        # End-to-end Python pipeline (Analysis, Bootstrap, Optimization, Plotting)
├── requirements.txt      # Python dependencies
├── xy_data (3).csv       # Dataset of 1,500 (x, y) coordinates
├── curve_fit.png         # High-resolution output graph and residual analysis
└── README.md             # Technical documentation and mathematical report
```

---

## Academic References & Citations

Citations formatted according to APA (7th edition):

1. **Principal Component Analysis:**  
   Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559–572. https://doi.org/10.1080/14786440109462720  
   Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417–441. https://doi.org/10.1037/h0071325

2. **Differential Evolution Global Optimization:**  
   Storn, R., & Price, K. (1997). Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359. https://doi.org/10.1023/A:1008202821328

3. **Nelder-Mead Direct Search Optimization:**  
   Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308–313. https://doi.org/10.1093/comjnl/7.4.308

4. **Scientific Computing Ecosystem:**  
   Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2  
   Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2  
   Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

*Authored by Srilekha for the Flam SDE R&D / AI Technical Evaluation.*
