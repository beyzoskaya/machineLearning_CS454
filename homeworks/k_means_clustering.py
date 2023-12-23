import csv
import zipfile
import math
import matplotlib.pyplot as plt
import os 
import random
import numpy as np

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            data.append([int(pixel) for pixel in row[1:]])
    return data

def custom_sum(values):
    result = 0
    for value in values:
        result += value
    return result

def euclidean_distance(point1, point2):
    squared_distance = 0
    for i in range(len(point1)):
        squared_distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(squared_distance)

def initialize_centroids(k, data):
     centroids = random.sample(data,k)
     return centroids

def assign_instances_to_centroids(data,centroids,k):
    num_instances = len(data)
    cluster_assignments = [0] * num_instances
    bi = [[0] * k for _ in range(num_instances)] 
    for i in range(num_instances):
        min_distance = float('inf')
        min_index = -1
        for j in range(k):
            distance = euclidean_distance(data[i], centroids[j])
            if distance < min_distance:
                min_distance = distance
                min_index = j
        bi[i] = [1 if idx == min_index else 0 for idx in range(k)]
        #print(f"bi value for {i}th index: {bi[i]}")
        cluster_assignments[i] = min_index
        #print(f"Cluster assignment for {i}th index: {cluster_assignments[i]}")

    return cluster_assignments, bi

def update_centroids_1(centroids, cluster_assignments, data, bi):
    k = len(centroids)
    num_instances = len(data)

    new_centroids = [[0 for _ in range(len(data[0]))] for _ in range(k)]

    for i in range(k):
        sum_weights = sum(bi[j][i] for j in range(num_instances))
        if sum_weights > 0:
            for f in range(len(centroids[i])):
                new_centroids[i][f] = sum(bi[j][i] * data[j][f] for j in range(num_instances)) / sum_weights
                #print(f"New centroid for {f}th feature: {new_centroids[i][f]}")
        else:
            new_centroids[i] = centroids[i]
            #print(f"Inside else block and the new centroid value of {i}th cluster: {new_centroids[i]}")

    return new_centroids

def k_means_clustering(data, k, max_iterations=10, convergence_threshold=1e-4):
    num_instances, num_features = len(data), len(data[0])

    centroids = initialize_centroids(k, data)
    # print("Initial centroids:", centroids)

    iteration = 0
    loss_per_iteration = []

    while iteration < max_iterations:
        cluster_assignments, bi = assign_instances_to_centroids(data, centroids, k)
        # print("Sample cluster assignments:", cluster_assignments[:10])

        # print(f"Centroids before iteration {iteration}:", centroids)
        new_centroids = update_centroids_1(centroids=centroids, cluster_assignments=cluster_assignments, bi=bi, data=data)
        # print(f"Centroids after iteration {iteration}:", centroids)

        # reconstruction_error = 0
        # for i in range(num_instances):
        #     for j in range(k):
        #         if bi[i][j] == 1:
        #             reconstruction_error += euclidean_distance(data[i], centroids[j]) ** 2

        reconstruction_error = 0
        for i in range(len(centroids)):
            for j in range(num_instances):
                if bi[j][i] == 1:
                    reconstruction_error += euclidean_distance(data[j], centroids[i]) ** 2


        # reconstruction_error = sum([bi[t][i] * euclidean_distance(data[t], centroids[i])**2 for t in range(num_instances) for i in range(len(centroids))])

        loss_per_iteration.append(reconstruction_error)

        iteration += 1
        print(f"Iteration {iteration}, Reconstruction Loss: {reconstruction_error}")
        centroids = new_centroids
    plot_loss(loss_per_iteration, k)

    # Print the final reconstruction loss
    print(f"Final Reconstruction Loss: {reconstruction_error}")

    return centroids, cluster_assignments, loss_per_iteration, bi

output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)

# Plot the cluster centroids as images during convergence
def plot_centroids_during_convergence(centroids, k):
    fig, axes = plt.subplots(2, k // 2, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(k):
        axes[i].imshow(np.array(centroids[i]).reshape(28, 28), cmap='gray')
        axes[i].axis('off')

    fig.suptitle('Centroids During Convergence')
    fig.savefig(os.path.join(output_folder, f'centroids_during_convergence_k_{k}.png'))
    plt.show()

def plot_centroids_after_convergence(centroids, k):
    fig, axes = plt.subplots(1, k, figsize=(k * 2, 2))  # Adjust the size as needed
    if k == 1:
        axes = [axes]  # Ensure axes is iterable for k=1

    for i in range(k):
        axes[i].imshow(np.array(centroids[i]).reshape(28, 28), cmap='gray')
        axes[i].axis('off')

    fig.suptitle(f'Centroids for k={k}')
    plt.show()
    fig.savefig(os.path.join(output_folder, f'centroids_after_convergence_k_{k}.png'))


def plot_loss(all_losses, k_values):
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']  
    if type(all_losses[0]) is list:
        for i, k in enumerate(k_values):
            plt.plot(range(len(all_losses[i])), all_losses[i], label=f'k={k}', color=colors[i % len(colors)])

        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Reconstruction Loss', fontsize=14)
        plt.title('Reconstruction Loss for Different k Values', fontsize=16)
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'reconstruction_error_for_k_values.png'))
        plt.show()
    else:
        k = k_values
        plt.plot(all_losses)
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Reconstruction Loss', fontsize=14)
        plt.title('Reconstruction Loss for Different k Values', fontsize=16)
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'reconstruction_error_for_k_{k}.png'))
        plt.show()
        


def main():
    zip_path = './fashion_mnist.zip'
    extract_path = './extracted_data'

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    csv_file_path = os.path.join(extract_path, 'fashion_mnist.csv')
    data = load_data(csv_file_path)

    k_values = [10, 20, 30]
    all_losses = []  # Accumulate loss values for different k

    for k in k_values:
        print(f"Running K-means clustering for k = {k}")
        centroids, _, loss, _ = k_means_clustering(data, k)

        # Plot the cluster centroids during convergence
        #plot_centroids_during_convergence(centroids, k)

        # Plot the cluster centroids after convergence
        plot_centroids_after_convergence(centroids, k)

        # Accumulate loss values for different k
        all_losses.append(loss)

    # Plot the reconstruction loss for all k values after the loop
    plot_loss(all_losses, k_values)

if __name__ == "__main__":
    main()
