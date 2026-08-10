# Probabilistic River Level Forecasting with LSTM and APU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch / TensorFlow](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange.svg)](#)

A confidence-aware deep learning framework for short- to medium-term probabilistic river-level forecasting using univariate historical stage time series. The system integrates **Long Short-Term Memory (LSTM)** networks with the **Analytical Propagation of Uncertainty (APU)** method to generate calibrated predictive distributions (2 to 48-hour horizons) without the computational overhead of sampling-based approaches.

---

## Table of Contents

- [About the Project](#about-the-project)
  - [Why it is Useful](#why-it-is-useful)
  - [Key Features](#key-features)
- [Methodology](#methodology)
  - [1. Deterministic LSTM Architecture](#1-deterministic-lstm-architecture)
  - [2. Analytical Propagation of Uncertainty (APU)](#2-analytical-propagation-of-uncertainty-apu)
  - [3. Risk-Informed Alert Framework](#3-risk-informed-alert-framework)
- [Study Area & Dataset](#study-area--dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Probabilistic Inference & APU Calculation](#probabilistic-inference--apu-calculation)
- [Experimental Results](#experimental-results)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## About the Project

Traditional hydrological models and deep learning frameworks often present deterministic point forecasts ($\mu_t$). However, in non-linear hydrological systems and data-scarce basins, a single point prediction without an explicit confidence interval is insufficient for critical decision-making during flood emergencies.

This repository provides an operational, real-time capable pipeline applied to the **Sinos River Basin** (São Leopoldo, Rio Grande do Sul, Brazil). By combining univariate LSTM neural networks with APU, the framework outputs continuous probability density functions, confidence intervals ($95\%$), and exact exceedance probabilities for operational flood thresholds.

### Why it is Useful

1. **Operates in Data-Scarce Settings:** Requires only historical river stage data (telemetric level time series) without relying on real-time precipitation or meteorological networks.
2. **Real-Time Efficiency:** APU calculates total uncertainty analytically using automatic differentiation and Taylor series expansion, avoiding slow sampling methods like Monte Carlo Dropout or Markov Chain Monte Carlo (MCMC).
3. **Actionable Alert Rules:** Incorporates multi-tiered alert triggers (*Watch*, *Warning*, *Emergency*), operational persistence criteria, and mathematical hysteresis loops to eliminate false alarms and "alert fatigue."

### Key Features

- **Univariate LSTM Backbone:** 50 hidden units, sliding window of $10$ time steps (2.5 hours of 15-min interval history).
- **Comprehensive Uncertainty Budget:** Explicitly accounts for sensor measurement noise ($\pm 0.5\%$ full scale, $ pprox 5	ext{ cm}$) and model residual variances (systematic and random error separation).
- **Hysteresis & Stability Mechanics:** Prevents rapid toggling between alert states during flood recessions.
- **Extreme Event Validation:** Evaluated on historical flood records, including the unprecedented events of May 2024 and June 2025.

---

## Methodology

The pipeline operates in two core stages:

```
+-------------------------------------------------------------------+
|                            PART I                                 |
|  [ Level Sensors ] -> [ Data Cleaning ] -> [ Univariate LSTM ]    |
|                                                     |             |
+-----------------------------------------------------|-------------+
                                                      v
+-------------------------------------------------------------------+
|                            PART II                                |
|  [ APU Calculation ] -> [ 95% Confidence Band ] -> [ Risk Alerts ]|
+-------------------------------------------------------------------+
```

### 1. Deterministic LSTM Architecture
- **Input Window:** $10$ lags ($2.5$ hours at 15-minute intervals).
- **Layers:** Single LSTM layer ($50$ units, `tanh` activation) + Dropout ($0.1$) + Dense layer ($1$ output neuron, linear activation).
- **Loss & Optimizer:** Mean Squared Error (MSE) with Adam optimizer.

### 2. Analytical Propagation of Uncertainty (APU)
Total predictive uncertainty $u_P(v_0)$ is derived from the combination of data uncertainty ($u_D$) and model uncertainty ($u_M$):

$$u_P^2(v_0) = u_D^2(v_0) + u_M^2$$

- **Data Uncertainty ($u_D$):** Propagates sensor calibration errors through the differentiable network via automatic differentiation matrices ($M_1, M_2, M_3, M_4$).
- **Model Uncertainty ($u_M$):** Computed from training residual standard deviation ($RE$).
- **Expanded Uncertainty ($U_P$):**

$$U_P(v_0) = k_p \cdot \sqrt{u_P^2(v_0)} + SE$$

*(where $k_p = 1.96$ for a 95% confidence interval and $SE$ is the systematic error bias).*

### 3. Risk-Informed Alert Framework
Predictive distribution parameters ($\mu_t, \sigma_t$) are evaluated against threshold $A$:

- **Exceedance Probability:** $p_t = \mathbb{P}(Y_t \ge A) = 1 - \Phi\left(rac{A - \mu_t}{\sigma_t}
ight)$
- **Watch:** $U_t \ge A$ or $p_t \ge 0.25$.
- **Warning:** $p_t \ge 0.50$ (or persistence of $U_t \ge A$ across consecutive steps).
- **Emergency:** Lower confidence bound exceeds threshold ($L_t \ge A$) or $p_t \ge 0.90$.
- **Hysteresis:** Alert downgrades require $p_t \le 0.20$ for $k = 2$ consecutive steps.

---

## Study Area & Dataset

- **Location:** Sinos River Basin, São Leopoldo station (Code: `87382000`, SGB-CPRM / ANA).
- **Coordinates:** Lat `-29.7589`, Lon `-51.1483`.
- **Sampling Interval:** 15-minute resolution telemetric water level series.
- **Alert Levels:** Attention ($358	ext{ cm}$), Alert ($395	ext{ cm}$), Emergency ($480	ext{--}500	ext{ cm}$).

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (Optional, for faster retraining)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/probabilistic-river-forecasting-apu.git
   cd probabilistic-river-forecasting-apu
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts ctivate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Data Preparation

To process raw sensor time series, clean missing values, and structure lagging matrices:

```bash
python src/data_preprocessing.py --input data/raw/sao_leopoldo_raw.csv --output data/processed/sinos_clean.csv
```

### Model Training

Train the univariate LSTM across temporal cross-validation folds (5-fold TimeSeriesSplit):

```bash
python src/train_lstm.py --data data/processed/sinos_clean.csv --epochs 30 --batch_size 32 --horizon 8
```

### Probabilistic Inference & APU Calculation

Generate multi-horizon forecasts with 95% uncertainty bands and risk alerts:

```bash
python src/evaluate_apu.py --model models/lstm_sinos.pth --test_data data/processed/june_2025_holdout.csv --sensor_error 5.0
```

---

## Experimental Results

Performance evaluation across different lead time horizons on the independent test set (June 2025 extreme event):

| Forecast Horizon | MAE (cm) | RMSE (cm) | $R^2$ Score | Operational Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **2 h** | 2.08 | 3.56 | 0.9994 | High precision |
| **4 h** | 4.23 | 6.73 | 0.9977 | High precision |
| **8 h** | 8.37 | 12.27 | 0.9924 | Optimal operational window |
| **12 h** | 13.84 | 40.67 | 0.9168 | Moderate accuracy |
| **24 h** | 16.76 | 26.30 | 0.9652 | Reliable trend indication |
| **48 h** | 36.95 | 115.82 | 0.3253 | High uncertainty band expansion |

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                 # Raw telemetric level readings
│   └── processed/           # Processed and scaled time series
├── docs/
│   ├── CONTRIBUTING.md      # Contribution guidelines
│   └── CODE_OF_CONDUCT.md   # Project code of conduct
├── models/                  # Trained neural network checkpoints
├── notebook/                # Jupyter Notebooks for EDA and visualization
├── src/
│   ├── apu_engine.py        # Analytical Propagation of Uncertainty core logic
│   ├── data_preprocessing.py # Cleaning, gap-filling, and min-max normalization
│   ├── decision_rules.py    # Triggers, hysteresis, and alert logic
│   ├── evaluate_apu.py      # Evaluation and plot generator
│   └── train_lstm.py        # Model definition and cross-validation training
├── LICENSE                  # Software license
├── README.md                # Repository documentation
└── requirements.txt         # Python dependency specifications
```

---

## Contributing

Contributions are welcome! Please read [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for details on our code of conduct and the submission process.

1. Fork the Repository
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this framework or methodology in your research, please cite our manuscript:

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
- **Institution:** Graduate Program in Applied Computing, Unisinos / SENAI Innovation Institute for Sensing Systems (ISI-SIM)
- **Email:** gersoneduardomello@gmail.com
