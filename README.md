# Contextual Bandit-based GNN for MCC Prediction


This repository contains the implementation of a Graph Variational Autoencoder (GVAE) and Graph Neural Network (GNN) model for Multiple Chronic Condition (MCC) Prediction using Contextual Bandit-based Graph Optimization. The framework processes synthetic patient datasets and generates a graph-based structure to improve prediction accuracy.

# Overview
This project implements a Graph Variational Autoencoder (GVAE) to reconstruct an adjacency matrix and node features from patient data, followed by a Graph Neural Network (GNN) trained with a Contextual Bandit (CB) algorithm for adaptive learning. The model iteratively optimizes the graph structure to enhance MCC classification.

# Key components:
- **GVAE for Graph Representation Learning.**
- **GNN for MCC Classification.**
- **Contextual Bandit Algorithm for Adaptive Learning.**


# Usage Instructions

Follow these steps to set up and run the project:

## **Step 1: Set Up the Environment**
You can create a **conda environment** using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate GVAE-GNN
```


If you prefer to install dependencies manually using pip, you can do:

```bash
pip install tensorflow keras numpy pandas networkx spektral matplotlib scipy scikit-learn
```


## **Step 2: Prepare the Data**
Ensure you have the dataset ready. The default script expects a CSV file named **`synthetic_dataset_general_mcc.csv`** in the project directory.

If using a different dataset, update the file path in:

```python
data = pd.read_csv('your_dataset.csv')
```

## **Step 2: Train the Model**

Run the main script to train the Graph Variational Autoencoder (GVAE) and Graph Neural Network (GNN):

```python
python train_model.py
```
### This process includes:

- **Training the GVAE to reconstruct adjacency matrices.**
- **Running the Contextual Bandit algorithm for adaptive learning.**
- **Evaluating the model on Accuracy, AUC-ROC, Precision, Recall, and F1-score.**
