import numpy as np

class kNearestNeighbors:
    """kNN Algorithm"""
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict_one(self, x_hat, ptype="classification"):
        neighbor_dists = []
        for x in self.X:
            neighbor_dists.append(np.linalg.norm(x_hat-x))

        nn = [y for _, y in sorted(zip(neighbor_dists, self.y))][:self.k]
        if ptype == 'classification':
            return max(nn, key=nn.count)
        elif ptype == 'regression':
            return sum(nn)/len(nn)

    def predict(self, X, ptype="classification"):
        pred = []
        for x_hat in X:
            pred.append(self.predict_one(x_hat, ptype))
        return np.array(pred)