import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
import os

# ✅ 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 경로 설정
base_path = "D:/PythonProject/Curtailment_Predictor/results"
y_val_path = os.path.join(base_path, "y_val.csv")
y_pred_path = os.path.join(base_path, "y_pred.csv")

# ✅ 데이터 불러오기
y_val = pd.read_csv(y_val_path)['value'].values
y_pred = pd.read_csv(y_pred_path)['value'].values

# ✅ MAPE 계산 (y_val == 0 제외)
mape_per_sample = np.zeros_like(y_val)
non_zero_mask = y_val != 0
mape_per_sample[non_zero_mask] = np.abs((y_val[non_zero_mask] - y_pred[non_zero_mask]) / y_val[non_zero_mask]) * 100
avg_mape = np.mean(mape_per_sample[non_zero_mask])

# ✅ 시각화
plt.figure(figsize=(14, 5))
plt.plot(mape_per_sample, color='orange', linewidth=1.5, label="샘플별 MAPE (%)")
plt.axhline(avg_mape, color='red', linestyle='--', linewidth=1.5, label=f"평균 MAPE: {avg_mape:.2f}%")
plt.title("시간 순서별 MAPE")
plt.xlabel("시점 index")
plt.ylabel("오차율 (%)")
plt.ylim(0, 500)  # ✅ y축 최대값 고정
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
