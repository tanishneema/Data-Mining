from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.metrics import pairwise_distances
import pandas as pd
import warnings
import time
from scipy.spatial.distance import cdist
import numpy as np

warnings.filterwarnings("ignore")

X = pd.read_csv("./Sem2/DM/kmeans_data/data.csv").values
y = pd.read_csv("./Sem2/DM/kmeans_data/label.csv").values.ravel()
K = len(np.unique(y))

def majorityVote(clusterLabel, trueLabel, k):
    mapp = {}
    for i in range(k):
        if np.any(clusterLabel == i):
            majorityLabel = Counter(trueLabel[clusterLabel == i]).most_common(1)[0][0]
            mapp[i] = majorityLabel
        else:
            mapp[i] = 0
    predictedLabel = np.vectorize(mapp.get)(clusterLabel)
    return np.mean(predictedLabel == trueLabel)

def euclideanKmeans(X, y, k, maxIter=500, stopType="centroid"):
    centroid = X[np.random.choice(X.shape[0], k, replace=False)]
    prevSSE = float('inf')
    for i in range(maxIter):
        distance = cdist(X, centroid, 'euclidean')
        cluster = np.argmin(distance, axis=1)
        newCentroid = np.array([X[cluster == j].mean(axis=0) if np.any(cluster == j) else centroid[j] for j in range(k)])
        SSE = sum(np.sum((X[cluster == j] - newCentroid[j]) ** 2) for j in range(k))
        if (stopType == "centroid" and np.allclose(centroid, newCentroid)) or \
           (stopType == "sse_increase" and SSE > prevSSE):
            break
        centroid = newCentroid
        prevSSE = SSE
    accu = majorityVote(cluster, y, k)
    return SSE, accu, i + 1

def cosine_kmeans(X, y, k, maxIter=500, stopType="centroid"):
    X = normalize(X)
    centroid = X[np.random.choice(X.shape[0], k, replace=False)]
    prevSSE = float('inf')
    for i in range(maxIter):
        distance = pairwise_distances(X, centroid, metric='cosine')
        cluster = np.argmin(distance, axis=1)
        newCentroid = np.array([normalize([X[cluster == j].mean(axis=0)])[0] if np.any(cluster == j) else centroid[j] for j in range(k)])
        SSE = sum(np.sum(pairwise_distances(X[cluster == j], [newCentroid[j]], metric='cosine') ** 2) for j in range(k))
        if (stopType == "centroid" and np.allclose(centroid, newCentroid)) or \
           (stopType == "sse_increase" and SSE > prevSSE):
            break
        centroid = newCentroid
        prevSSE = SSE
    accu = majorityVote(cluster, y, k)
    return SSE, accu, i + 1

def jaccard_kmeans(X, y, k, maxIter=500, stopType="centroid"):
    X_bin = (X > X.mean(axis=0)).astype(int)
    centroid = X_bin[np.random.choice(X.shape[0], k, replace=False)]
    prevSSE = float('inf')
    for i in range(maxIter):
        distance = pairwise_distances(X_bin, centroid, metric="jaccard")
        cluster = np.argmin(distance, axis=1)
        newCentroid = np.array([np.round(X_bin[cluster == j].mean(axis=0)).astype(int) if np.any(cluster == j) else centroid[j] for j in range(k)])
        SSE = sum(np.sum(pairwise_distances(X_bin[cluster == j], [newCentroid[j]], metric="jaccard") ** 2) for j in range(k))
        if (stopType == "centroid" and np.array_equal(centroid, newCentroid)) or \
           (stopType == "sse_increase" and SSE > prevSSE):
            break
        centroid = newCentroid
        prevSSE = SSE
    accu = majorityVote(cluster, y, k)
    return SSE, accu, i + 1

for stopType in ["centroid", "sse_increase", "maxIter"]:
    print(f"\nStop Condition: {stopType}")

    start = time.time()
    SSEEuclidean, accuEuclidean, iterEuclidean = euclideanKmeans(X, y, K, 100 if stopType == "maxIter" else 500, stopType)
    eu_time = time.time() - start

    start = time.time()
    SSECosine, accuCosine, iterCosine = cosine_kmeans(X, y, K, 100 if stopType == "maxIter" else 500, stopType)
    cosTime = time.time() - start

    start = time.time()
    SSEJaccard, accuJaccard, iterJaccard = jaccard_kmeans(X, y, K, 100 if stopType == "maxIter" else 500, stopType)
    jac_time = time.time() - start

    print("Euclidean SSE:", SSEEuclidean, "Accuracy:", accuEuclidean, "Iterations:", iterEuclidean, "Time:", round(eu_time, 4))
    print("Cosine SSE:", SSECosine, "Accuracy:", accuCosine, "Iterations:", iterCosine, "Time:", round(cosTime, 4))
    print("Jaccard SSE:", SSEJaccard, "Accuracy:", accuJaccard, "Iterations:", iterJaccard, "Time:", round(jac_time, 4))
