import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load data from CSV file
file_path = "your_file.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Ensure the CSV has the required columns
if 'x' not in data.columns or 'y' not in data.columns:
    raise ValueError("CSV file must contain 'x' and 'y' columns.")

# Step 2: Extract x and y values
X = data[['x', 'y']].values

# Step 3: Perform k-means clustering
k = 3  # Number of clusters, adjust as needed
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

# Step 4: Get cluster centers
centroids = kmeans.cluster_centers_

# Step 5: Visualize the results
plt.figure(figsize=(8, 6))
for cluster_id in range(k):
    cluster_data = data[data['cluster'] == cluster_id]
    plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster_id}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('K-Means Clustering')
plt.show()

# Step 6: Print centroids
print("Centroids of clusters:")
print(centroids)
