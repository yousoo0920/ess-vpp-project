import pandas as pd
import numpy as np

def evaluate_mape(data_path, true_col='출력제한량', pred_col='예측출력제한량'):
    df = pd.read_csv(data_path)

    # 0인 실제값 때문에 분모 0 방지용 마스킹
    mask = df[true_col] != 0
    df = df[mask]

    # MAPE 계산
    mape = np.mean(np.abs((df[true_col] - df[pred_col]) / df[true_col])) * 100
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    return mape
