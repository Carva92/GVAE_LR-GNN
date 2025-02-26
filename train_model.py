def totEdge(n) :
 
    result = (n * (n - 1)) // 2
 
    return result
             

if __name__ == "__main__" :
 
    n = 1592
 
    print(totEdge(n))


import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

print('Start Code', flush=True)

import sys
import scipy
import random
import numpy as np
import pandas as pd
import keras
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from spektral.layers import GCNConv
from scipy.sparse import csr_matrix
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
from networkx.algorithms import bipartite
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import OneHotEncoder
from spektral.utils import normalized_adjacency
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LinearRegression


import tensorflow_memory_utility as tfu
import data_preparation as dp


def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    np.random.seed(1)
    random.seed(1)
    
    
reset_random_seeds()
print(tf.config.list_physical_devices('GPU'))
print(tf.__version__,flush=True)

# Read the CSV file
data = pd.read_csv('synthetic_dataset_general_mcc.csv')
patients = data[['RRID', 'RISK_F1']]
conditions_features = data[['RISK_F2', 'RISK_F3', 'RISK_F4', 'RISK_F5', 'RISK_F6', 'RISK_F7', 'RISK_F8', 'RISK_F9', 'RISK_F10', 'RISK_F11', 'RISK_F12', 'RISK_F13', 'RISK_F14']]


df = dp.e_distance(conditions_features, patients)
data_new = data.iloc[:, 2:15].copy()
data_new.index = data_new.index + 1

# create graph from edge data
G = nx.from_pandas_edgelist(df, source='PatientX', target='PatientY', edge_attr='Euclidean_Dist')

# add node features
for node_id, features in data_new.iterrows():
    # convert the features for each node into a dictionary
    feature_dict = features.to_dict()
    # add these features to the corresponding node in the graph
    G.nodes[node_id].update(feature_dict)

A_n = nx.adjacency_matrix(G, weight='Euclidean_Dist')
dense_matrix = A_n.toarray()


A = nx.adjacency_matrix(G, weight='Euclidean_Dist')
features = [
    [node[1]['RISK_F2'], node[1]['RISK_F3'], node[1]['RISK_F4'], 
     node[1]['RISK_F5'], node[1]['RISK_F6'], node[1]['RISK_F7'], 
     node[1]['RISK_F8'], node[1]['RISK_F9'], node[1]['RISK_F10'], 
     node[1]['RISK_F11'], node[1]['RISK_F12'], node[1]['RISK_F13'], node[1]['RISK_F14']]
    for node in G.nodes(data=True)
]


arr = np.array(features)
edges_arr = list(G.edges)
edges = map(np.array, edges_arr)
edges = np.array(list(edges)).T
edge_weights = tf.convert_to_tensor(df['Euclidean_Dist'])
edge_weights = tf.cast(edge_weights, tf.float32)


# For each of the last three columns
for i in range(10, 13):  # Columns 10, 11, 12 have index 8, 9, 10 respectively
    col_min, col_max = arr[:, i].min(), arr[:, i].max()
    
    # Define categories thresholds
    threshold1, threshold2 = col_min + (col_max - col_min) / 3, col_min + 2 * (col_max - col_min) / 3
    
    # Assign categories
    arr[(arr[:, i] >= col_min) & (arr[:, i] < threshold1), i] = 1
    arr[(arr[:, i] >= threshold1) & (arr[:, i] < threshold2), i] = 2
    arr[arr[:, i] >= threshold2, i] = 3


# Let's say columns at index 0, 1, 4, 5, 7, 8, 9, 10 are multi-class categorical columns
categorical_features_indices = [0, 1, 4, 5, 7, 8, 9, 10]

#encoder = OneHotEncoder(sparse=False)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(arr[:, categorical_features_indices])

# Transform only the categorical columns
one_hot_encoded_features = encoder.transform(arr[:, categorical_features_indices])

# Concatenate the one-hot encoded columns with the original array
X_encoded = np.hstack((arr[:, [i for i in range(arr.shape[1]) if i not in categorical_features_indices]], one_hot_encoded_features))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X_encoded)


# Convert adjacency matrix and features to TensorFlow tensors
A_norm = normalized_adjacency(csr_matrix(A))
A_tensor = tf.convert_to_tensor(A_norm.toarray(), dtype=tf.float32)
X_tensor = tf.convert_to_tensor(X,   dtype=tf.float32)



# Convert string categories to integers
unique_categories = list(set(data["CATEGORY"]))
category_dict = {cat: i for i, cat in enumerate(unique_categories)}
y = np.array([category_dict[cat] for cat in data["CATEGORY"]], dtype=np.int32)
y_tensor = tf.convert_to_tensor(y)
y_tensor = tf.cast(y_tensor, dtype=tf.int32)


# Ensure we've loaded the entire dataset into these tensors
assert y_tensor.shape[0] == len(patients)
assert X_tensor.shape == (len(patients), 34)
assert A_tensor.shape == (len(patients), len(patients))


# Loss functions
def loss_fn(logits):
    labels = tf.ones_like(logits)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

def gnn_loss_fn(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def kl_divergence(mean, variance):
    """Calculate KL divergence with direct variance (ensured positive)."""
    # Ensuring variance is positive
    variance = tf.clip_by_value(variance, 1e-7, np.inf)
    return -0.5 * tf.reduce_sum(1 + tf.math.log(variance) - tf.square(mean) - variance)


# Helper function to remove or add edges based on their importance and accuracy change
def adjust_edges_by_importance(reconstructed_adj_np, k, increase_accuracy):
    # Flatten the matrix and get indices
    flattened_weights = reconstructed_adj_np.flatten()
    sorted_indices = np.argsort(flattened_weights)
    
    # Depending on accuracy increase or decrease, remove or add edges
    k = int(k)
    print(k,flush=True)
    for i in range(k):
        idx = sorted_indices[i] if increase_accuracy else sorted_indices[-(i+1)]
        row, col = np.unravel_index(idx, reconstructed_adj_np.shape)
        if increase_accuracy:
            reconstructed_adj_np[row, col] = 0
            reconstructed_adj_np[col, row] = 0
        else:
            reconstructed_adj_np[row, col] = 1
            reconstructed_adj_np[col, row] = 1
        
    return reconstructed_adj_np


# Encoder Layer
class Encoder(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(2 * out_channels)
        self.conv2 = GCNConv(out_channels)
        self.mean_layer = tf.keras.layers.Dense(latent_dim, name='Mean')
        self.variance_layer = tf.keras.layers.Dense(latent_dim, name='Variance')

    def call(self, inputs, training=False):
        x, a = inputs
        x = tf.keras.activations.relu(self.conv1([x, a]))
        x = self.conv2([x, a])
        
        mean = self.mean_layer(x)
        variance = self.variance_layer(x)
        
        epsilon = tf.random.normal(shape=mean.shape, mean=0.0, stddev=1.0)
        z = mean + tf.math.softplus(variance) * epsilon
        return z, mean, variance

# Decoder Layer
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_features):
        super(Decoder, self).__init__()
        self.feature_decoder = tf.keras.layers.Dense(num_features, activation='relu')

    def call(self, z):
        reconstructed_adj = tf.matmul(z, z, transpose_b=True)
        reconstructed_features = self.feature_decoder(z)
        return reconstructed_adj, reconstructed_features

# Graph Autoencoder (GAE)
class GAE(tf.keras.Model):
    def __init__(self, in_channels, out_channels, latent_dim, num_features):
        super(GAE, self).__init__()
        self.encoder = Encoder(in_channels, out_channels, latent_dim)  # add latent_dim here
        self.decoder = Decoder(num_features)
        self.latent_dim = latent_dim


    def call(self, inputs):
        x, a = inputs
        z, mean, variance = self.encoder([x, a])

        # Assuming z is the concatenation of mean and log_variance.
        kl_loss = kl_divergence(mean, variance)

        # Sample from the distribution
        epsilon = tf.random.normal(shape=mean.shape)
        mean = tf.clip_by_value(mean, clip_value_min=-10, clip_value_max=10)
        safe_variance = tf.clip_by_value(variance, clip_value_min=-10, clip_value_max=10)
        sampled_z = mean + tf.exp(0.5 * safe_variance) * epsilon

        #sampled_z = mean + tf.exp(0.5 * variance) * epsilon

        reconstructed_adj, reconstructed_features = self.decoder(sampled_z)
        
        return sampled_z, reconstructed_adj, reconstructed_features, kl_loss


num_classes = len(unique_categories)


def create_ffn(hidden_units, dropout_rate, name=None, regularizer=None):
    ffn_layers = []
    for units in hidden_units:
        ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))
        ffn_layers.append(tf.keras.layers.Dense(units, activation=tf.nn.gelu, kernel_regularizer=regularizer))
    return tf.keras.Sequential(ffn_layers, name=name)



def split_graph_variant(adjacency_matrix, features_matrix, labels, test_size=0.4):
    """
    Splits the graph into training and testing sets.

    :param adjacency_matrix: Adjacency matrix of the graph (numpy array).
    :param features_matrix: Feature matrix of the nodes (numpy array).
    :param labels: Labels for each node (numpy array).
    :param test_size: Fraction of the graph to be used as test data.
    :return: A tuple containing training and testing adjacency matrices, feature matrices, and labels.
    """
    num_nodes = adjacency_matrix.shape[0]
    node_indices = np.arange(num_nodes)

    # Split node indices into training and testing sets
    # If stratification is not possible, perform a simple split
    train_indices, test_indices = train_test_split(node_indices, test_size=test_size, random_state=42)

    train_adj = adjacency_matrix[np.ix_(train_indices, train_indices)]
    test_adj = adjacency_matrix[np.ix_(test_indices, test_indices)]

    train_features = features_matrix[train_indices]
    test_features = features_matrix[test_indices]

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    return train_adj, test_adj, train_features, test_features, train_labels, test_labels




class GNN(tf.keras.Model):
    def __init__(self, graph_info, num_classes, hidden_units, dropout_rate=0.4, regularizer='l2'):
        super(GNN, self).__init__()
        node_features, edges, edge_weights = graph_info
        
       
        # Regularization strategy
        if regularizer == 'l2':
            reg = regularizers.l2(0.01)
        elif regularizer == 'l1':
            reg = regularizers.l1(0.01)
        else:
            reg = None 
        
        
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights if edge_weights is not None else tf.ones(shape=edges.shape[1])
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess", regularizer=reg)
        self.conv1 = GCNConv(32, activation='relu', kernel_regularizer=reg)  # Applying regularization to GCNConv if it supports
        self.conv2 = GCNConv(16, activation='relu', kernel_regularizer=reg)
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess", regularizer=reg)
        self.compute_logits = tf.keras.layers.Dense(units=num_classes, activation=None, name="logits", kernel_regularizer=reg)

    def call(self, inputs):
        x, a = inputs
        x = self.preprocess(x)
        x = tf.keras.activations.relu(self.conv1([x, a]))
        x = self.conv2([x, a])
        x = self.postprocess(x)
        return self.compute_logits(x)


class MyModel(tf.keras.Model):
    def __init__(self, base_model, lambda_=0.001):
        super(MyModel, self).__init__()
        self.base_model = base_model
        self.lambda_ = lambda_
        self.dense = tf.keras.layers.Dense(695)  # Adjust as needed

    def call(self, inputs):
        x, A, D = inputs
        y_pred = self.base_model(x)

        # Adjust y_pred if necessary
        y_pred = self.dense(y_pred)
        y_pred = tf.expand_dims(y_pred, -1)

        L = D - A
        y_pred_vertices = tf.matmul(L, y_pred)
        laplacian_loss = tf.reduce_mean(tf.square(y_pred - y_pred_vertices))
        self.add_loss(self.lambda_ * laplacian_loss)

        return y_pred

# Example of creating the model
hidden_units = [64,64]
dropout_rate = 0.02
graph_info = (X_tensor, edges, edge_weights)  # Define these accordingly
base_model = GNN(graph_info, num_classes, hidden_units, dropout_rate)
gnn_model = MyModel(base_model)

         
                  
                  
gnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])



def create_gnn_model(graph_info, num_classes, hidden_units, dropout_rate):
    gnn_model = GNN(graph_info, num_classes, hidden_units, dropout_rate)
    gnn_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003)
    return gnn_model, gnn_optimizer


def create_gae_model(input_dim, hidden_dim, latent_dim, feature_dim):
    gae_model = GAE(input_dim, hidden_dim, latent_dim, feature_dim)
    gae_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003, clipvalue=0.5)
    return gae_model, gae_optimizer


def compute_k(base_k, accuracy):
    delta = accuracy - 0.5
    new_k = int(base_k * (1 + delta))
    return new_k


best_accuracy = 0
base_k = 15000
num_iterations = 20
acc = 0
stored_graphs = []
accuracies = []
gae_loss_values = []
gnn_loss_values = []
gae_loss_values_per_iteration = []
gnn_loss_values_per_iteration = []
selected_accuracy = 0


# Define the path for saving the best model's weights
best_gnn_weights_path = 'path/best_gnn_weights_new.h5'
best_gae_model_path = 'path/best_gae_model.h5'
# Ensure the directory exists
os.makedirs(os.path.dirname(best_gnn_weights_path), exist_ok=True)
os.makedirs(os.path.dirname(best_gae_model_path), exist_ok=True)



unique_categories = len(np.unique(y_tensor.numpy()))


def generate_graph_variants(reconstructed_adj, reconstructed_features, num_variants=10):
    variants = []
    for i in range(num_variants):
        # Define noise levels
        noise_level_adj = 0.05  # Adjust as needed

        # Generate masks for non-zero values
        mask_adj = reconstructed_adj != 0

        # Add Gaussian noise only to non-zero elements of the adjacency matrix
        adj_noise = np.random.normal(0, noise_level_adj, reconstructed_adj.shape) * mask_adj

        # Create the adjacency matrix variant by adding noise
        adj_variant = reconstructed_adj + adj_noise

        # Ensure symmetry and zero diagonal for adjacency matrix
        adj_variant = np.triu(adj_variant, 1) + np.triu(adj_variant, 1).T
        np.fill_diagonal(adj_variant, 0)

        # Keep the features unaltered
        features_variant = reconstructed_features

        # Store the variant
        variants.append((adj_variant, features_variant))
    return variants



def extract_features(variant):
    adj_variant, features_variant = variant

    # Graph-based features
    num_edges = np.sum(adj_variant != 0) / 2  # Each edge is counted twice in an undirected graph
    num_nodes = adj_variant.shape[0]
    graph_density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    avg_degree = 2 * num_edges / num_nodes

    # Node features-based statistics
    mean_features = np.mean(features_variant)
    std_features = np.std(features_variant)

    # Concatenate all features into a single vector (ensure this matches your feature_dimension)
    combined_features = np.hstack([num_edges, num_nodes, graph_density, avg_degree, mean_features, std_features])

    # If combined_features length is less than feature_dimension, you might need to adjust or fill with zeros
    if combined_features.shape[0] < feature_dimension:
        # Extend the combined_features to match the expected dimension
        combined_features = np.pad(combined_features, (0, feature_dimension - combined_features.shape[0]), 'constant')

    return combined_features




import time
import numpy as np
import tensorflow as tf
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Initialization of variables and settings for the contextual bandit approach
start_time = time.time()  # Record the start time of the training
training_output = []  # This will store your log messages
best_accuracy = 0.0
selected_variant_accuracies = []
num_variants = 10  # Adjust based on how many graph variants you generate each iteration
stored_graphs = []
gae_loss_values_per_iteration = []
gnn_loss_values_per_iteration = []

# Contextual bandit model initialization
feature_dimension = 3481
alpha_value = 0.05  # You might need to experiment with different values
contextual_bandit_model = Lasso(alpha=alpha_value)
X_bandit = np.zeros((0, feature_dimension))  # Initialize as empty, with correct feature dimension
y_bandit = np.zeros(0)  # Initialize as empty
model_fitted = False  # Indicator to check if the model has been fitted

with tf.device('/GPU:0'):
    for iteration in range(num_iterations):
        model, optimizer = create_gae_model(X_tensor.shape[-1], 64, 64, X_tensor.shape[-1])
        gae_loss_values = []
        gnn_loss_values = []
        accuracies = []  # Store accuracies of all variants for the current iteration

        # 1. Train Graph Autoencoder
        for epoch in range(3000):
            with tf.GradientTape() as tape:
                _, reconstructed_adj, reconstructed_features, _ = model([X_tensor, A_tensor])
                features_loss = tf.losses.mean_squared_error(X_tensor, reconstructed_features)
                adj_loss = tf.losses.mean_squared_error(A_tensor, reconstructed_adj)
                total_loss = features_loss + adj_loss
                gae_loss_values.append(total_loss.numpy())

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if epoch % 500 == 0:  # Log the process
                message = f"Iteration: {iteration + 1}, GAE Epoch: {epoch}, Loss: {total_loss.numpy()[0]:.4f}"
                print(message)
                training_output.append(message)

        _, reconstructed_adj, reconstructed_features, _ = model([X_tensor, A_tensor])
        reconstructed_adj_np = reconstructed_adj.numpy()
        np.fill_diagonal(reconstructed_adj_np, 0)

        # Use adjust_edges_by_importance to adjust the reconstructed adjacency matrix
        k = compute_k(base_k, selected_accuracy)
        reconstructed_adj_np = adjust_edges_by_importance(reconstructed_adj_np, k, best_accuracy)

        variants = generate_graph_variants(reconstructed_adj_np, reconstructed_features.numpy())
        variants.insert(0, (reconstructed_adj_np, reconstructed_features.numpy()))

        selected_accuracies = []
        variant_accuracies = []
        # Inside the iteration where variants are tested
        variant_auc_scores = []   # Store AUC scores for the current iteration
        variant_precisions = []   # Store Precision scores for the current iteration
        variant_recalls = []      # Store Recall scores for the current iteration
        variant_f1_scores = []    # Store F1 scores for the current iteration


        # Iterate through graph variants
        for variant_index, (variant_adj, variant_features) in enumerate(variants):
            # Split the variant into training and testing sets
            train_adj, test_adj, train_features, test_features, train_labels, test_labels = split_graph_variant(variant_adj, variant_features, y)

            # Create the GNN model
            graph_info = (train_features, edges, edge_weights)  # Ensure these are correctly defined for training data
            gnn, gnn_optimizer = create_gnn_model(graph_info, unique_categories, hidden_units, dropout_rate)

            # Train the GNN model on the training set
            for epoch in range(7000):
                with tf.GradientTape() as tape:
                    logits = gnn([train_features, train_adj])
                    classification_loss = gnn_loss_fn(logits, train_labels)
                    if gnn.losses:
                        total_loss = classification_loss + tf.add_n(gnn.losses)
                    else:
                        total_loss = classification_loss
                    gnn_loss_values.append(total_loss.numpy())
                    grads = tape.gradient(classification_loss, gnn.trainable_variables)
                    gnn_optimizer.apply_gradients(zip(grads, gnn.trainable_variables))

                    if epoch % 500 == 0:  # Every 500 epochs
                        pred = tf.argmax(logits, axis=1, output_type=tf.int32)
                        train_accuracy = np.mean(np.where(train_labels == pred, 1, 0))
                        message = f"Iteration: {iteration + 1}, GNN Epoch: {epoch}, Loss: {classification_loss.numpy():.4f}, Training Accuracy: {train_accuracy:.4f}"
                        print(message)
                        training_output.append(message)

            # Evaluate the model on the testing set after training
            test_logits = gnn([train_features, train_adj])
            test_pred = tf.argmax(test_logits, axis=1, output_type=tf.int32)
            
            import torch
            import torch.nn.functional as F
            
            # Assuming test_logits are the raw logits from your model
            # test_logits = torch.tensor(test_logits)  # If it's not already a tensor

            # Convert logits to probabilities using softmax
            test_probs = tf.nn.softmax(test_logits, axis=1).numpy()  # Apply softmax across classes
            
            # New Revision
            
            #test_pred_np = test_pred.numpy()
            #test_labels_np = train_labels.numpy()
            
            
            
            test_accuracy = np.mean(np.where(train_labels == test_pred, 1, 0))
            
            #print(test_labels.shape, flush = True)  # Should be (n_samples,)
            #print(test_probs.shape, flush = True)   # Should be (n_samples, n_classes)
                  
            # Compute AUC (Area Under Curve) - assumes binary or multilabel classification problem
            if len(np.unique(train_labels)) == 2:  # AUC score is valid for binary classification
                auc_score = roc_auc_score(train_labels, test_logits.numpy()[:, 1])  # Assuming logits for class 1
            else:
                auc_score = roc_auc_score(train_labels, test_probs, average='weighted', multi_class='ovr')
            
            # Compute Precision, Recall, and F1 Score
            precision = precision_score(train_labels, test_pred, average='weighted', zero_division=1)
            recall = recall_score(train_labels, test_pred, average='weighted', zero_division=1)
            f1 = f1_score(train_labels, test_pred, average='weighted', zero_division=1)
            
            # Log the results for the current variant
            print(f'Variant {variant_index + 1}:')
            print(f' - Test Accuracy: {test_accuracy:.4f}')
            print(f' - AUC Score: {auc_score:.4f}')
            print(f' - Precision: {precision:.4f}')
            print(f' - Recall: {recall:.4f}')
            print(f' - F1 Score: {f1:.4f}')
            
            # Store the metrics for later use
            variant_accuracies.append(test_accuracy)
            variant_auc_scores.append(auc_score)
            variant_precisions.append(precision)
            variant_recalls.append(recall)
            variant_f1_scores.append(f1)
            
            
            
            

            # Log and use the test_accuracy for further processing or decision-making
            print(f'Variant {variant_index + 1}: Test Accuracy: {test_accuracy:.4f}')
            variant_accuracies.append(test_accuracy)

        # Contextual bandit selection process
        if iteration > 0:
            contextual_bandit_model.fit(X_bandit, y_bandit)
        contexts = np.array([extract_features(variant) for variant in variants])

        if model_fitted:
            expected_rewards = contextual_bandit_model.predict(contexts)
        else:
            expected_rewards = np.random.rand(len(variants))

        selected_index = np.argmax(expected_rewards)
        selected_accuracy = variant_accuracies[selected_index]

        X_bandit = np.vstack([X_bandit, contexts[selected_index]])
        y_bandit = np.append(y_bandit, selected_accuracy)

        
        min_samples_to_fit = 10
        
        if X_bandit.shape[0] > min_samples_to_fit:
            contextual_bandit_model.fit(X_bandit, y_bandit)
            model_fitted = True

        #message = f"Iteration {iteration + 1}: Selected Variant {selected_index + 1}, Expected Reward: {expected_rewards[selected_index]:.4f}, Actual Test Accuracy: {selected_accuracy:.4f}"
        #print(message, flush=True)
        #training_output.append(message)
        
        
        message = f"Iteration {iteration + 1}: Selected Variant {selected_index + 1}, "
        message += f"Expected Reward: {expected_rewards[selected_index]:.4f}, "
        message += f"Actual Test Accuracy: {selected_accuracy:.4f}, "
        message += f"AUC: {variant_auc_scores[selected_index]:.4f}, "
        message += f"Precision: {variant_precisions[selected_index]:.4f}, "
        message += f"Recall: {variant_recalls[selected_index]:.4f}, "
        message += f"F1: {variant_f1_scores[selected_index]:.4f}"
        
        # Log the message
        print(message, flush=True)
        training_output.append(message)

        
        
        selected_graph_adj, selected_graph_features = variants[selected_index]
        G_reconstructed = nx.from_numpy_array(selected_graph_adj)
        stored_graphs.append(G_reconstructed)

        selected_variant_accuracies.append(selected_accuracy)
        if selected_accuracy > best_accuracy:
            best_accuracy = selected_accuracy

            # Save the GAE model
            model.save_weights(best_gae_model_path)
            print(f"Saved best GAE model to {best_gae_model_path}")
            
            # Save the weights of the GNN model
            gnn.save_weights(best_gnn_weights_path)
            print(f"Saved best GNN model weights to {best_gnn_weights_path}")

        gae_loss_values_per_iteration.append(gae_loss_values)
        gnn_loss_values_per_iteration.append(gnn_loss_values)
        gae_loss_values = []  # Reset for the next iteration
        gnn_loss_values = []  # Reset for the next iteration

# Output final results
message = f"Best accuracy: {best_accuracy:.4f} achieved in graph iteration: {selected_variant_accuracies.index(best_accuracy) + 1}"
print(message, flush=True)
training_output.append(message)
end_time = time.time()  # Record the end time of the training
total_training_time = end_time - start_time  # Calculate the total training time
training_output.append(f"Total training time: {total_training_time:.2f} seconds")


# Output the best performing metrics
best_auc = max(variant_auc_scores)
best_precision = max(variant_precisions)
best_recall = max(variant_recalls)
best_f1 = max(variant_f1_scores)

message = f"Best AUC: {best_auc:.4f} achieved in graph iteration: {variant_auc_scores.index(best_auc) + 1}"
print(message, flush=True)
training_output.append(message)

message = f"Best Precision: {best_precision:.4f} achieved in graph iteration: {variant_precisions.index(best_precision) + 1}"
print(message, flush=True)
training_output.append(message)

message = f"Best Recall: {best_recall:.4f} achieved in graph iteration: {variant_recalls.index(best_recall) + 1}"
print(message, flush=True)
training_output.append(message)

message = f"Best F1 Score: {best_f1:.4f} achieved in graph iteration: {variant_f1_scores.index(best_f1) + 1}"
print(message, flush=True)
training_output.append(message)


# Specify the folder where you want to save the graphs
folder_path = 'path'

# Create the folder if it doesn't already exist
os.makedirs(folder_path, exist_ok=True)

# Construct the full path for the output file
output_file_path = f'{folder_path}/training_output.txt'


# Write the training output to the file
with open(output_file_path, 'w') as file:
    for item in training_output:
        file.write("%s\n" % item)



for idx, graph in enumerate(stored_graphs):
    print(graph.number_of_edges(), flush=True)
    
    
    
    
def get_config(self):
    config = super(YourLayer, self).get_config()
    config.update({
        "custom_arg1": self.custom_arg1,
        "custom_arg2": self.custom_arg2,
        # Include all initialization arguments
    })
    return config
    
    
model_config = {
    'model_type': 'GAE',
    'in_channels': X_tensor.shape[-1],
    'out_channels': 64,
    'latent_dim': 64,
    'feature_dim': X_tensor.shape[-1],
    # Include any other relevant configuration details
}

import json
with open(best_gae_model_path + '_config.json', 'w') as json_file:
    json.dump(model_config, json_file)
    
    
    
with open(best_gae_model_path + '_config.json', 'r') as json_file:
    model_config = json.load(json_file)

# Assume `model_config` has been loaded as shown in your previous step

# Recreate the model
model = GAE(model_config['in_channels'], model_config['out_channels'], 
            model_config['latent_dim'], model_config['feature_dim'])

# Prepare a sample input with the correct shape
# Note: Adjust `sample_input` to match the shape expected by your model's first layer
sample_input_shape = (1, model_config['in_channels'])  # Example: (batch_size, feature_dim)
sample_adj_shape = (1, 1)  # Adjust based on your adjacency matrix shape
sample_input = tf.random.normal(sample_input_shape)
sample_adj = tf.random.normal(sample_adj_shape)

# Call the model with the sample input
# This step initializes the model's weights
#_ = model([sample_input, sample_adj])

# Now, load the weights
#model.load_weights('Test4/best_gae_model.h5')    
    


import os
import pickle



for idx, graph in enumerate(stored_graphs):
    # Construct a unique file name for each graph, including the folder path
    filename = f'{folder_path}/Graph{idx + 1}.pickle'  # Graph numbering starts from 1
    
    # Serialize and save the graph object to file
    with open(filename, 'wb') as file:
        pickle.dump(graph, file)






