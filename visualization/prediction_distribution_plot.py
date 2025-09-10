import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform

# ✅ 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 데이터 불러오기
def load_data():
    y_val = pd.read_csv("D:/PythonProject/Curtailment_Predictor/results/y_val.csv")['value']
    y_pred = pd.read_csv("D:/PythonProject/Curtailment_Predictor/results/y_pred.csv")['value']
    return y_val.values, y_pred.values

# ✅ 분포 비교 히스토그램
def plot_distribution(y_true, y_pred):
    plt.figure(figsize=(8, 5))
    plt.hist(y_true, bins=40, alpha=0.6, label='실제값', color='black', edgecolor='white')
    plt.hist(y_pred, bins=40, alpha=0.5, label='예측값', color='blue', edgecolor='white')
    plt.title("예측값 vs 실제값 분포 비교")
    plt.xlabel("출력제한량")
    plt.ylabel("빈도수")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ✅ 실행
if __name__ == "__main__":
    y_val, y_pred = load_data()
    plot_distribution(y_val, y_pred)
    input("Press Enter to exit...")
