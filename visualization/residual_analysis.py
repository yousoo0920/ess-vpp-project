import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ 데이터 불러오기
def load_data():
    y_val = pd.read_csv("D:/PythonProject/Curtailment_Predictor/results/y_val.csv")['value']
    y_pred = pd.read_csv("D:/PythonProject/Curtailment_Predictor/results/y_pred.csv")['value']
    return y_val.values, y_pred.values

# ✅ 잔차 시각화
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    # 시계열 그래프
    plt.figure(figsize=(12, 4))
    plt.plot(residuals, label="Residual", color='blue')
    plt.title("Residual Over Time")
    plt.xlabel("Time Index")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 히스토그램
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ✅ 성능 지표 출력
def print_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)

    print("📊 모델 성능 지표")
    print(f"✅ MAE  : {mae:.4f}")
    print(f"✅ RMSE : {rmse:.4f}")
    print(f"✅ R²   : {r2:.4f}")
    print(f"✅ MAPE : {mape:.2f}%")

# ✅ MAPE 계산 함수 (0 제외)
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


    print("📊 모델 성능 지표")
    print(f"✅ MAE  : {mae:.4f}")
    print(f"✅ RMSE : {rmse:.4f}")
    print(f"✅ R²   : {r2:.4f}")
    print(f"✅ MAPE : {mape:.2f}%")

# ✅ 메인 실행
if __name__ == "__main__":
    y_val, y_pred = load_data()
    print_metrics(y_val, y_pred)
    plot_residuals(y_val, y_pred)
    input("Press Enter to exit...")
