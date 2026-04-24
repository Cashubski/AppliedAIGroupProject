# Applied AI Group Project

## Setup and Installation

We recommend using a virtual environment to isolate the project dependencies.

1. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the virtual environment:**
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Install the required packages:**
   The project dependencies are listed in the `requirements.txt` file. You can install them using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Project Tasks and Notebooks

The solutions to the different project tasks are divided into the following standalone Jupyter notebooks:

- **Baseline**: `MMAC_Baseline_executed.ipynb`
- **Class Imbalance Solution**: `MMAC_Imbalance.ipynb`
- **Multi-task learning**: `MMAC_MTL.ipynb`
- **Explainability markers**: `MMAC_Explainability_Markers.ipynb`
- **Uncertainty evaluation**: `MMAC_Uncertainty.ipynb`
- **Bias Solution**: `MMAC_Bias.ipynb`
- **Mixed solution**: `MMAC_Mixed.ipynb`

## Utility Scripts

- **`mmac_utils.py`**: This Python script contains utility functions and helper code shared across the `MMAC_MTL.ipynb` and `MMAC_Uncertainty.ipynb` notebooks.