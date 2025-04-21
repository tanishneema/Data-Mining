import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate, KFold
from surprise.accuracy import mae, rmse
import matplotlib.pyplot as plt

file_path = "./Sem2/DM/archive/ratings_small.csv"
df = pd.read_csv(file_path)

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

kf = KFold(n_splits=5)

models = {
    "PMF": SVD(biased=False),
    "User_CF": KNNBasic(sim_options={'userBased': True, 'name': 'msd'}),
    "Item_CF": KNNBasic(sim_options={'userBased': False, 'name': 'msd'})
}

results = {}

for name, model in models.items():
    maeScores = []
    rmseScores = []
    for trainset, testset in kf.split(data):
        model.fit(trainset)
        predic = model.test(testset)
        maeScores.append(mae(predic, verbose=False))
        rmseScores.append(rmse(predic, verbose=False))
    results[name] = {
        "MAE": sum(maeScores) / len(maeScores),
        "RMSE": sum(rmseScores) / len(rmseScores)
    }

for model, metrics in results.items():
    print(f"{model} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

# Q2: Plot MAE and RMSE for comparison
modelList = list(results.keys())
maeValue = [results[model]['MAE'] for model in modelList]
rmseValue = [results[model]['RMSE'] for model in modelList]

x = range(len(modelList))
plt.figure(figsize=(10, 5))
plt.bar(x, maeValue, width=0.4, label='MAE', align='center')
plt.bar([i + 0.4 for i in x], rmseValue, width=0.4, label='RMSE', align='center')
plt.xticks([i + 0.2 for i in x], modelList)
plt.ylabel('Error')
plt.title('MAE and RMSE Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("q2_mae_rmse_plot.png")
plt.close()

# Q6: Neighbor Impact
def evalNeighbors(userBased):
    ks = [10, 20, 30, 40, 50]
    rmses = []
    for k in ks:
        algo = KNNBasic(k=k, sim_options={'name': 'msd', 'userBased': userBased})
        rmseScores = []
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            predic = algo.test(testset)
            rmseScores.append(rmse(predic, verbose=False))
        rmses.append(sum(rmseScores) / len(rmseScores))
    bestK = ks[rmses.index(min(rmses))]
    print(f"Best K for {'User-Based' if userBased else 'Item-Based'} CF: {bestK}")
    return ks, rmses

ksUser, rmsesUser = evalNeighbors(userBased=True)
ksItem, rmsesItem = evalNeighbors(userBased=False)

plt.figure(figsize=(10, 5))
plt.plot(ksUser, rmsesUser, marker='o', label='User-Based CF')
plt.plot(ksItem, rmsesItem, marker='s', label='Item-Based CF')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Average RMSE')
plt.title('Impact of Neighbors on RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("q6_neighbors_vs_rmse.png")
plt.close()

# Q5: Similarity Metric Impact
def evalSimilarity(sim_name, userBased):
    algo = KNNBasic(sim_options={'name': sim_name, 'userBased': userBased})
    rmseScores = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predic = algo.test(testset)
        rmseScores.append(rmse(predic, verbose=False))
    return sum(rmseScores) / len(rmseScores)

similarity = ['cosine', 'msd', 'pearson']
userRmseSim = [evalSimilarity(sim, True) for sim in similarity]
itemRmseSim = [evalSimilarity(sim, False) for sim in similarity]

x = range(len(similarity))
plt.figure(figsize=(10, 5))
plt.bar(x, userRmseSim, width=0.4, label='User-Based CF', align='center')
plt.bar([i + 0.4 for i in x], itemRmseSim, width=0.4, label='Item-Based CF', align='center')
plt.xticks([i + 0.2 for i in x], similarity)
plt.ylabel('Average RMSE')
plt.title('RMSE Comparison by Similarity Metric')
plt.legend()
plt.tight_layout()
plt.savefig("q5_similarity_comparison.png")
plt.close()
