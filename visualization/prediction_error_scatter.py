import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ✅ MAPE 산점도 시각화
def plot_instance_mape(y_true, y_pred):
    # 인스턴스별 MAPE (%)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    mape_per_instance = np.zeros_like(y_true)
    mape_per_instance[non_zero] = np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]) * 100

    mape_avg = np.mean(mape_per_instance[non_zero])

    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(len(mape_per_instance)), mape_per_instance, s=3, alpha=0.5, color='blue')
    plt.axhline(mape_avg, color='red', linewidth=2, label=f'평균 MAPE = {mape_avg:.2f}%')
    plt.title("Instances vs MAPE")
    plt.xlabel("Instances (샘플 인덱스)")
    plt.ylabel("MAPE (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ✅ 실행
if __name__ == "__main__":
    y_val, y_pred = load_data()
    plot_instance_mape(y_val, y_pred)
    input("Press Enter to exit...")
