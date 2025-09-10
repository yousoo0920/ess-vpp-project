import numpy as np

def create_sequences(data, input_steps=3, target_col='is_curtailment'):
    X, y = [], []
    values = data['wind_speed'].values
    targets = data[target_col].values

    for i in range(len(values) - input_steps):
        X.append(values[i:i+input_steps])
        y.append(targets[i+input_steps])  # 다음 시점 예측

    return np.array(X), np.array(y)
