# Custom implementation and comparison of 3 fundamental machine learning algorithms on MATLAB

## Project Overview
This project compares the performance of three machine learning algorithms (Adaline, Logistic Regressor, and Perceptron) in solving binary classification problems. The analysis is conducted on three subsets of the Iris dataset:

## Repository Structure
- **`/models/`**: MATLAB function files for each model.
  - `AdalineNeuron.m`
  - `LogisticRegressorFunction.m`
  - `PerceptronFunction.m`
- **`main_code.m`**: Main script for running the experiments.
- **`/data/`**: # Input data files.
  - **`subset3.mat`**: Marginally linearly separable data.
  - **`Iris.dat`**: Consists of all the 150 samples. 
- **`/results/`**: Output results and visualizations.
  - **`evaluation_metrics.csv`**
  - **`mse_visualizations/`**
  - **`decision_boundaries/`**
- **`README.md`**: Project overview and instructions.
- **`LICENSE`**

### Iris dataset:
1. **Subset #1**: Linearly separable data, manually preprocessed in the main code.
2. **Subset #2**: Non-linearly separable data, manually preprocessed in the main code.
3. **Subset #3**: Marginally linearly separable synthetic data, provided as a separate file (subset3.mat).

The project aims to:
* Evaluate the Mean Squared Error (MSE) and classification accuracy across subsets.
* Compare model performance on linearly separable, non-linearly separable, and marginally separable data.
* Visualize decision boundaries for the best and worst models.

## Features
* **Custom Implementation:**
  * Adaline (Linear Neuron)
  * Logistic Regression
  * Perceptron
* **10-Fold Cross-Validation:**
  * Maximum of 50 epochs per experiment.
  * Learning rates: 0.1, 0.05, and 0.01.

## Data
### Subsets
1. **Subset #1**: Two species from the Iris dataset (samples 1-100), encoded as +1/-1 using the 1st and 3rd features. This subset is manually preprocessed in the main code.
2. **Subset #2**: Non-linearly separable subset of the Iris dataset (samples 51-150), encoded as +1/-1. This subset is manually preprocessed in the main code.
3. **Subset #3**: Synthetic marginally linearly separable subset derived from Subset #1, provided as a separate file (`subset3.mat`).

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/tSopermon/comparison-ml-algorithms.git
   cd comparison-ml-algorithms
   ```
2. Open MATLAB and ensure all necessary files are in the working directory.
3. Run the `main_code.m` script:
   ```bash
   main_code
   ```

## Results
### Metrics
1. **Mean Squared Error (MSE):**
    * Average across 10 folds.
    * Plots for the best and worst models for each algorithm and learning rate.
2. **Classification Accuracy:
    * Average accuracy on control data.
    * Best and worst model accuracy.

### Visualizations:
1. **MSE History:**
    * For each subset: 18 plots (6 per model for best and worst models).
2. **Decision Boundaries:**
    * Training and control sets with boundaries for best and worst models.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact:
- **Author**: Nikolaos Theokritos Tsopanidis
- **Email**: nikos.tsopanidis746@gmail.com
