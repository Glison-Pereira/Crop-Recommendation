
# Crop Recommendation System

This repository contains a machine learning project for recommending crops based on soil and climatic conditions using a Decision Tree Classifier.

## Overview

The goal of this project is to predict the best crop to grow based on several environmental factors such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall. The model is trained on historical crop recommendation data to provide accurate predictions for new inputs.

## Contents

- `Crop_recommendation.csv`: The dataset used for training and testing the model.
- `croprecommendation.py`: The main script containing the code for data preprocessing, model training, evaluation, and prediction.

## Setup

### Prerequisites

Ensure you have the following Python packages installed:

- `pandas`
- `scikit-learn`

You can install these packages using pip:

```bash
pip install pandas scikit-learn
```

### Running the Script

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Crop-Recommendation.git
   cd Crop-Recommendation
   ```

2. **Place the dataset in the repository:**

   Ensure `Crop_recommendation.csv` is in the same directory as `croprecommendation.py`.

3. **Run the script:**

   Execute the script using Python:

   ```bash
   python croprecommendation.py
   ```

4. **Provide Inputs:**

   The script will prompt you to enter values for:
   - Nitrogen
   - Phosphorus
   - Potassium
   - Temperature
   - Humidity
   - pH
   - Rainfall

   After entering these values, the model will predict the best crop for the given conditions.

## Code Explanation

1. **Data Loading and Cleaning**:
   - Load the dataset and drop unnecessary columns.
   - Handle missing and duplicate data.

2. **Feature and Target Preparation**:
   - Separate features and target variables.
   - Scale the features using Min-Max Scaling.

3. **Model Training**:
   - Split the data into training and testing sets.
   - Train a Decision Tree Classifier on the training data.

4. **Model Evaluation**:
   - Evaluate the model using confusion matrix, classification report, and accuracy score.

5. **Prediction**:
   - Use the trained model to make predictions based on user inputs.

## Example Usage

```python
Enter nitrogen: 50
Enter phosphorus: 30
Enter potassium: 40
Enter temperature: 25
Enter humidity: 70
Enter pH: 6
Enter rainfall: 100
```

The model will output the recommended crop based on the provided inputs.

