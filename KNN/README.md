# Customer Segmentation Analysis: From Scratch

This project implements **K-Nearest Neighbors (K-NN)** for classification and explores the dataset structure using manual implementations (without using Scikit-Learn). It is designed to demonstrate an understanding of distance-based machine learning algorithms and data visualization.

## Project Overview
* **Goal:** Classify customers into segments (0 or 1) based on their `Annual_Income_k` and `Spending_Score`.
* **Technique:** K-Nearest Neighbors (K-NN) algorithm implemented from scratch using NumPy.
* **Environment:** Python, NumPy, Pandas, Matplotlib.

## Implementation Details

### 1. K-NN Algorithm
The K-NN implementation calculates the Euclidean distance between a target customer and all points in the training set:
$$\text{Distance} = \sqrt{\sum (x_{test} - x_{train})^2}$$
It then identifies the $K$ nearest neighbors and performs a majority vote to predict the class label.

### 2. Data Preparation
* The dataset is split into **80% training data** and **20% testing data** using a custom shuffling function.
* The data is treated as a 2D coordinate system where features represent X and Y axes.

## Results
The model successfully classifies test points by analyzing their proximity to known labeled clusters.

**Visualization:**
The graph below depicts the training clusters (Blue: Class 0, Orange: Class 1) and the test customer (Red Star) being classified.

![K-NN Classification Result](knn_visualization.png) *(Note: Ensure you rename your saved image to match this filename)*

**Prediction Accuracy:**
* **Prediction:** 1
* **Actual Label:** 1.0

## How to Run
1.  Ensure you have the required libraries installed: `pip install numpy pandas matplotlib`.
2.  Place `customer_segmentation_data.csv` in the same directory as the script.
3.  Run the main script: `python your_script_name.py`.

---

### Tips for your submission:
* **The "From Scratch" Aspect:** Make sure to highlight in your report that you **did not use** `sklearn.neighbors` or `sklearn.model_selection`. This is the most important part of your professor's requirements!
* **Explain the Choice of K:** In your report, mention why you chose `K=5` (e.g., "An odd number was chosen to prevent ties in the majority vote").