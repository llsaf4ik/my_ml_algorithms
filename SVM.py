import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


class SVM:
    def __init__(self, alpha, epoch=100, lr=0.001):
        self.alpha = alpha
        self.epoch = epoch
        self.lr = lr

    def fit(self, X, y):
        self.w = np.random.normal(0, 1, X.shape[1])
        self.b = np.random.normal(0, 1)
        for _ in range(self.epoch):
            for i, x in enumerate(X):
                M = (np.dot(x, self.w) - self.b) * y[i]
                dw = self.alpha * self.w if M >= 1 else self.alpha * self.w - y[i] * x
                db = 0 if M >= 1 else y[i]
                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)


np.random.seed(17)
n = 500
X = pd.DataFrame({"x1": list(np.random.normal(50, 10, n)) + list(np.random.normal(30, 30, n)),
                "x2": list(np.random.normal(50, 10, n)) + list(np.random.normal(170, 50, n))})
X = (X-X.mean())/X.std()
y = np.array([1]*n + [-1]*n)


svm = SVM(alpha=0.001)
svm.fit(X.to_numpy(), y)
w1, w2 = svm.w
b = svm.b
print((svm.predict(X)==y).mean().round(3)*100)
print(X[~(svm.predict(X)==y)].reset_index(drop=True))

x1, x2 = X["x1"].mean() - X["x1"].std(), X["x1"].mean() + X["x1"].std()

X["target"] = y
print([-w1/w2, b/w2])
sns.scatterplot(data=X, x="x1", y="x2", hue=y, linewidth=0, palette="Set2")
plt.plot([x1, x2], [-w1/w2*(x1)+b/w2, -w1/w2*(x2)+b/w2])
#plt.axis("equal")
plt.grid(True)
plt.show()

