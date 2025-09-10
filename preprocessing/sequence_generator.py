import numpy as np

def create_sequences(X, y, time_steps=5):  # ✅ 반드시 5로
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)
