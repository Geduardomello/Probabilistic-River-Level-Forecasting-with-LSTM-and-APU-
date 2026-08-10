# Probabilistic River Level Forecasting with LSTM and APU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch / TensorFlow](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange.svg)](#)

A **confidence-aware deep learning framework** designed for real-time, probabilistic river level forecasting and risk management. This repository implements the **Analytical Propagation of Uncertainty (APU)** method integrated with Long Short-Term Memory (LSTM) networks. 

Instead of relying on rigid, deterministic point predictions ($\mu_t$), this framework converts deep learning outputs into **calibrated 95% uncertainty bands**, computes **exact threshold exceedance probabilities ($p_{t+h}$)**, and executes an **operational decision engine** (Watch, Warning, Emergency) with persistence and hysteresis filters (as demonstrated in **Figure 9** of the manuscript).

---

## Table of Contents

- [Core Focus: Analytical Propagation of Uncertainty (APU)](#core-focus-analytical-propagation-of-uncertainty-apu)
  - [Why Deterministic Forecasts Fail in Hydrology](#why-deterministic-forecasts-fail-in-hydrology)
  - [The APU Mathematical Framework](#the-apu-mathematical-framework)
  - [95% Prediction Bands & Uncertainty Budget](#95-prediction-bands--uncertainty-budget)
- [Operational Risk Engine & Decision Matrix (Figure 9)](#operational-risk-engine--decision-matrix-figure-9)
  - [Probabilistic Alert Triggers](#probabilistic-alert-triggers)
  - [Hysteresis and Stability Mechanics](#hysteresis-and-stability-mechanics)
  - [Detailed Hourly Decision Case Study](#detailed-hourly-decision-case-study)
- [Methodology & Pipeline](#methodology--pipeline)
- [Study Area & Dataset](#study-area--dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. LSTM Training](#2-lstm-training)
  - [3. APU Probabilistic Inference & Alert Generation](#3-apu-probabilistic-inference--alert-generation)
- [Experimental Results & Lead-Time Decay](#experimental-results--lead-time-decay)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Core Focus: Analytical Propagation of Uncertainty (APU)

### Why Deterministic Forecasts Fail in Hydrology
Single point estimates ($\mu_t$) convey false precision. In dynamic river basins like the **Sinos River**, small errors in stage inputs propagate nonlinearly. An uncalibrated point forecast predicting a stage 20 cm below a dike top gives a false sense of safety. Conversely, **APU equips decision-makers with continuous probability density functions (PDFs)**, quantifying exactly *how confident* the network is at every hour.

### The APU Mathematical Framework
Unlike sampling methods (Monte Carlo Dropout, Ensemble MCMC) which require hundreds of forward passes and hinder real-time deployment, **APU calculates predictive uncertainty analytically in a single forward pass** using automatic differentiation and Taylor series expansion:

1. **Total Predictive Variance:**
   $$u_P^2(v_0) = u_D^2(v_0) + u_M^2$$

2. **Data Uncertainty ($u_D$):** Propagates sensor noise through the network layers via matrix operations ($M_1 \circ M_2 \circ M_3$):
   $$u_D^2(v_0) = \sum_{i=1}^k \left(  rac{\partial y}{\partial x_i}(v_0) \cdot u(x_i) 
ight)^2 + 2 \sum_{i=1}^{k-1} \sum_{j=i+1}^k \left(  rac{\partial y}{\partial x_i}(v_0)  rac{\partial y}{\partial x_j}(v_0) 	ext{cov}(x_i, x_j) 
ight) + u^2(y)$$

3. **Model Uncertainty ($u_M$):** Separates systematic bias ($SE$) and residual variability ($RE$):
   $$SE =  rac{1}{n}\sum_{i=1}^n (F(x_i) - y_i), \quad RE^2 =  rac{1}{n}\sum_{i=1}^n [(F(x_i) - y_i) - SE]^2, \quad u_M = RE$$

### 95% Prediction Bands & Uncertainty Budget
The expanded predictive uncertainty $U_P(v_0)$ uses a coverage factor $k_p = 1.96$ to yield calibrated upper ($U_{t+h}$) and lower ($L_{t+h}$) confidence bounds:

$$U_P(v_0) = k_p \cdot \sqrt{u_P^2(v_0)} + SE \implies [L_{t+h}, U_{t+h}] = [\mu_{t+h} - U_P, \mu_{t+h} + U_P]$$

- **Sensor Noise Budget:** Evaluated conservatively at $\pm 0.5\%$ full scale ($ pprox \pm 5	ext{ cm}$), conforming to IEC 61298-2.

---

## Operational Risk Engine & Decision Matrix (Figure 9)

The core operational innovation lies in converting APU confidence bands $[L_{t+h}, U_{t+h}]$ into exact exceedance probabilities ($p_{t+h}$) against critical flood thresholds (e.g., $A = 480	ext{ cm}$):

$$p_{t+h} = \mathbb{P}(Y_{t+h} \ge A) = 1 - \Phi\left(  rac{A - \mu_{t+h}}{\sigma_{t+h}} 
ight), \quad 	ext{where } \sigma_{t+h}  pprox  rac{U_{t+h} - L_{t+h}}{2 \cdot 1.96}$$

```text
Forecast Evolution (8h Ahead) with Triggers, Hysteresis and Risk Communication
      Water Level (cm)
       520 |                                   *---* [Emergency]
       510 |                             *---*              500 |                       *---*              *---* [Warning* via Hysteresis]
  ---  480 | - - - - - - - - - * - - - - - - - - - - - - - -  < Alert Threshold (A = 480 cm)
       470 |             *---*
       460 |       *---* [Watch]
           +---------------------------------------------------> Time (Hours)
            0   1   2    3   4   5   6   7   8   9   10  11  12
```

### Probabilistic Alert Triggers
- **Watch:** Triggered when $U_{t+h} \ge A$ or $p_{t+h} \ge 0.25$. Activates internal monitoring.
- **Warning:** Issued when $p_{t+h} \ge 0.50$ (or persistence of $U_{t+h} \ge A$). Requires civil defense mobilization.
- **Emergency:** Declared when lower bound $L_{t+h} \ge A$ (100% operational guarantee of flood) or $p_{t+h} \ge 0.90$.
- **Hysteresis Loop:** Prevents "alert fatigue" during stage recession. Downgrading from *Warning* to *Watch* requires $p_{t+h} \le 0.20$ for $k = 2$ consecutive steps.

### Detailed Hourly Decision Case Study
Below is the exact reproduction of the 8-hour lead-time ($t+8$) decision sequence illustrated in **Figure 9** of the paper:

| Hour ($t$) | $\mu_{t+8}$ (cm) | $95\%$ Band $[L_{t+8}, U_{t+8}]$ | $p_{t+8}$ | Operational Decision | Key Driver / Mechanism |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 470 | $[450, 490]$ | 0.16 | **Watch** | Upper bound $U \ge 480	ext{ cm}$, low $p$ |
| **1** | 475 | $[455, 495]$ | 0.31 | **Watch** | Moderate risk expansion |
| **2** | 478 | $[458, 498]$ | 0.42 | **Watch** | Persistence increases risk |
| **3** | 485 | $[465, 505]$ | 0.69 | **Warning** | $p_{t+8} \ge 0.50$ threshold crossed |
| **4** | 490 | $[470, 510]$ | 0.84 | **Warning** | High flood probability |
| **5** | 495 | $[475, 515]$ | 0.93 | **Emergency** | $p_{t+8} \ge 0.90$ critical threshold |
| **6** | 498 | $[478, 518]$ | 0.96 | **Emergency** | Severe threat level |
| **7** | 500 | $[480, 520]$ | 0.98 | **Emergency** | **Operational Guarantee ($L_{t+8} \ge 480	ext{ cm}$)** |
| **8** | 498 | $[478, 518]$ | 0.96 | **Emergency** | Risk persistence |
| **9** | 490 | $[470, 510]$ | 0.84 | **Warning** | Stage receding, probability drops |
| **10** | 485 | $[465, 505]$ | 0.69 | **Warning** | Still within active risk zone |
| **11** | 478 | $[458, 498]$ | 0.42 | **Warning\*** | **Hysteresis blocks premature downgrade** |
| **12** | 475 | $[455, 495]$ | 0.31 | **Warning\*** | Downgrade delayed ($k=2$ steps requirement) |

---

## Methodology & Pipeline

```text
+-----------------------------------------------------------------------------------+
| PART I: DETERMINISTIC BASE                                                        |
| Telemetric Sensors (15-min) -> Gap Filling -> MinMaxScaler -> Univariate LSTM     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| PART II: APU UNCERTAINTY & RISK ENGINE                                            |
| Automatic Diff (M1-M4 Matrices) + Model Residuals -> Calibrated 95% Band [L, U]   |
|                                         |                                         |
|                                         v                                         |
| Calculate Exceedance Probabilities (p_t) -> Apply Hysteresis -> Issue Public Alert|
+-----------------------------------------------------------------------------------+
```

---

## Study Area & Dataset

- **Location:** Sinos River Basin, S√£o Leopoldo station (Code: `87382000`, SGB-CPRM / ANA).
- **Coordinates:** Lat `-29.7589`, Lon `-51.1483`.
- **Sampling Interval:** 15-minute resolution.
- **Key Reference Events:** Evaluated against major real-world floods, including May 2024 ($> 8.0	ext{ m}$) and June 2025 holdout events.

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch or TensorFlow

### Installation

```bash
git clone https://github.com/username/probabilistic-river-forecasting-apu.git
cd probabilistic-river-forecasting-apu
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts ctivate
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preprocessing
```bash
python src/data_preprocessing.py --input data/raw/sao_leopoldo.csv --output data/processed/sinos_clean.csv
```

### 2. LSTM Training
```bash
python src/train_lstm.py --data data/processed/sinos_clean.csv --epochs 30 --batch_size 32
```

### 3. APU Probabilistic Inference & Alert Generation
Run the APU engine to generate 95% bands, exceedance probabilities, and decision logs (Figure 9 format):

```bash
python src/evaluate_apu.py   --model models/lstm_sinos.pth   --test_data data/processed/june_2025_holdout.csv   --alert_threshold 480   --sensor_error 5.0
```

---

## Experimental Results & Lead-Time Decay

Performance and APU confidence interval expansion across lead times:

| Horizon | MAE (cm) | RMSE (cm) | $R^2$ Score | 95% Uncertainty Band Width | Operational Utility |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **2 h** | 2.08 | 3.56 | 0.9994 | Narrow ($\pm 7	ext{ cm}$) | Immediate flash flood warning |
| **4 h** | 4.23 | 6.73 | 0.9977 | Narrow ($\pm 13	ext{ cm}$) | Urban evacuation prep |
| **8 h** | 8.37 | 12.27 | 0.9924 | Moderate ($\pm 25	ext{ cm}$) | **Optimal operational decision window** |
| **12 h** | 13.84 | 40.67 | 0.9168 | Expanding ($\pm 50	ext{ cm}$) | Early resource staging |
| **24 h** | 16.76 | 26.30 | 0.9652 | Wide ($\pm 70	ext{ cm}$) | Regional trend assessment |
| **48 h** | 36.95 | 115.82 | 0.3253 | Very Wide ($> 150	ext{ cm}$) | High long-term uncertainty |

---

## Repository Structure

```text
.
‚îú‚îÄ‚îÄ data/
‚îÇ   ‚îú‚îÄ‚îÄ raw/                 # Telemetric sensor readings
‚îÇ   ‚îî‚îÄ‚îÄ processed/           # Filtered, scaled time series
‚îú‚îÄ‚îÄ docs/
‚îÇ   ‚îú‚îÄ‚îÄ CONTRIBUTING.md      # Contribution guidelines
‚îÇ   ‚îî‚îÄ‚îÄ CODE_OF_CONDUCT.md   # Project code of conduct
‚îú‚îÄ‚îÄ models/                  # Saved LSTM weights
‚îú‚îÄ‚îÄ notebook/                # Visualizations & APU step-by-step math
‚îú‚îÄ‚îÄ src/
‚îÇ   ‚îú‚îÄ‚îÄ apu_engine.py        # Core APU uncertainty propagation matrices
‚îÇ   ‚îú‚îÄ‚îÄ decision_rules.py    # Probabilistic triggers, persistence & hysteresis
‚îÇ   ‚îú‚îÄ‚îÄ evaluate_apu.py      # Figure 9 decision table & band generator
‚îÇ   ‚îî‚îÄ‚îÄ train_lstm.py        # Univariate LSTM cross-validation trainer
‚îú‚îÄ‚îÄ LICENSE                  # MIT License
‚îú‚îÄ‚îÄ README.md                # Repository documentation
‚îî‚îÄ‚îÄ requirements.txt         # Dependencies
```

---

## Contributing

Contributions are welcome! Please check [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) before submitting Pull Requests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

```bibtex
@article{mello2025probabilistic,
  title={Probabilistic River Level Forecasting with LSTM and APU: a Confidence-Aware Deep Learning Framework Applied to the Sinos River},
  author={de Mello, Gerson Eduardo and da Rosa Righi, Rodrigo and Cagliari, Joice},
  journal={Applied Computing and Geosciences},
  year={2025}
}
```

---

## Contact

- **Author:** Gerson Eduardo de Mello
- **Institution:** Graduate Program in Applied Computing (Unisinos) / SENAI Innovation Institute for Sensing Systems (ISI-SIM)
- **Email:** gersoneduardomello@gmail.com
