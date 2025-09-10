import pandas as pd
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
y_val_df = pd.read_csv(y_val_path)
y_pred_df = pd.read_csv(y_pred_path)

# ✅ 인덱스를 시간축으로 사용 (예: 일자 순서 or 샘플 인덱스)
index = range(len(y_val_df))

# ✅ 시계열 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(index, y_val_df['value'], label='실제 출력제한량', linewidth=2)
plt.plot(index, y_pred_df['value'], label='예측 출력제한량', linestyle='--', linewidth=2)
plt.xlabel("시간 순서 (샘플 인덱스)")
plt.ylabel("출력제한량 (MWh)")
plt.title("시간 순서 기반 예측 vs 실제 시계열 그래프")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
