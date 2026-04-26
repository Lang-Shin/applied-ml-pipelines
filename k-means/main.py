import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("customer_segmentation_data.csv", usecols=["Annual_Income_k","Spending_Score"])

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))

CLUSTERS = 2

dynamic_df = df
cluster_df = pd.DataFrame(columns=['1','2',"cluster"])
#initialize centroids at index 50 and 51
centroids = {"center1": np.array(df.iloc[50]), "center2": np.array(df.iloc[51])} 

previous_cluster = []


while True:
    print(f"current centroid: {centroids['center1']}, {centroids['center2']}")
    #populate/update the distance and cluster group table
    for row in dynamic_df.itertuples():
        d1 = euclidean_distance([row.Annual_Income_k, row.Spending_Score], centroids["center1"])
        d2 = euclidean_distance([row.Annual_Income_k, row.Spending_Score], centroids["center2"])

        cluster_df.loc[row.Index] = d1, d2, 1 if d1 < d2 else 2

    if previous_cluster == list(cluster_df["cluster"]):
        print('terminating loop')
        break

    #initialize prev cluster to new cluster to be used compared
    previous_cluster = list(cluster_df["cluster"])

    #get indices where cluster 11 and 2 are to recalculate centroid
    indices_1 = np.where(cluster_df["cluster"] == 1)[0].tolist()
    indices_2 = np.where(cluster_df["cluster"] == 2)[0].tolist()


    #for the ff, get the new centroid by adding up the clustered points and getting the average
    new_centroid_1 = []
    new_centroid_2 = []
    x1 = 0.0
    y1 = 0.0
    x2 = 0.0
    y2 = 0.0

    for i in indices_1:
        x1 += dynamic_df.iloc[i, 0]
    for i in indices_1:
        y1 += dynamic_df.iloc[i, 1]

    for i in indices_2:
        x2 += dynamic_df.iloc[i, 0]
    for i in indices_2:
        y2 += dynamic_df.iloc[i, 1]

    new_centroid_1 = [x1 / len(indices_1), y1 / len(indices_1)]
    new_centroid_2 = [x2 / len(indices_2), y2 / len(indices_2)]

    centroids["center1"] = np.array(new_centroid_1)
    centroids["center2"] = np.array(new_centroid_2)

    
#PLOTTING
colors = np.where(dynamic_df.index.isin(indices_1), 'red', 'blue')

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("K-Means Clustering", fontsize=15, fontweight="bold")
 
ax.scatter(
    dynamic_df['Annual_Income_k'], dynamic_df['Spending_Score'],
    c=colors, s=60, alpha=0.8, edgecolors="white", linewidths=0.5
)
ax.scatter(
    centroids["center1"][0], centroids["center1"][1],
    c="red",
    s=220, marker="X", edgecolors="black", linewidths=1.2, zorder=5
)
ax.scatter(
    centroids["center2"][0], centroids["center2"][1],
    c="blue",
    s=220, marker="X", edgecolors="black", linewidths=1.2, zorder=5
)

ax.set_xlabel("annual income")
ax.set_ylabel("spending score")
ax.set_title(f"Cluster Assignments  (k={CLUSTERS})")
ax.grid(alpha=0.3)
 
plt.tight_layout()
plt.show()
