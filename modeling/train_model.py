import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Regression import run_regression

# ✅ 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ MAPE 계산 함수
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# ✅ CSV 경로 및 피처 설정
csv_path = "D:/PythonProject/Curtailment_Predictor/data/processed_dataset_with_engineering.csv"
feature_columns = [
    "전일_출력제한량",
    "출력제한_변화율",
    "전일_태양광",
    "전일_풍력",
    "출력비율"
]

# ✅ 모델 학습 및 예측
df, model, scaler, y_pred, y_val = run_regression(
    csv_path=csv_path,
    feature_columns=feature_columns
)

# ✅ 결과 저장 경로 생성
save_dir = "D:/PythonProject/Curtailment_Predictor/results"
os.makedirs(save_dir, exist_ok=True)

# ✅ 예측 결과 저장
pd.DataFrame({'value': y_val}).to_csv(os.path.join(save_dir, "y_val.csv"), index=False)
pd.DataFrame({'value': y_pred}).to_csv(os.path.join(save_dir, "y_pred.csv"), index=False)

# ✅ 지표 계산
mape = mean_absolute_percentage_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"✅ MAPE: {mape:.2f}%")
print(f"✅ MAE : {mae:.2f}")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R²  : {r2:.4f}")

# ✅ 시각화
plt.figure(figsize=(9, 6))
plt.scatter(y_val, y_pred, alpha=0.6, label='예측값')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='y = x')
plt.xlabel("실제 출력제한량")
plt.ylabel("예측 출력제한량")
plt.title("실제값 vs 예측값 산점도")
plt.grid(True)
plt.legend(loc='upper left')

# 📌 지표 박스 위치 조정 (오른쪽 아래)
text_x = y_val.max() * 0.88
text_y = y_val.max() * 0.1
textstr = (
    f"MAPE: {mape:.2f}%\n"
    f"MAE : {mae:.2f}\n"
    f"RMSE: {rmse:.2f}\n"
    f"R²  : {r2:.4f}"
)
plt.text(
    text_x, text_y,
    textstr,
    fontsize=14,
    color='black',
    bbox=dict(facecolor='white', edgecolor='black')
)

plt.tight_layout()
plt.show()

# ✅ scaler 저장
import joblib  # 상단에 이거 추가

# 마지막 줄에 이거 추가 (또는 확인)
joblib.dump(scaler, "D:/PythonProject/Curtailment_Predictor/modeling/scaler.pkl")
print("✅ scaler 저장 완료")

# train_model.py 실행 마지막 부분에 추가
model.save("D:/PythonProject/Curtailment_Predictor/modeling/model_fixed.h5")
joblib.dump(scaler, "D:/PythonProject/Curtailment_Predictor/modeling/scaler_fixed.pkl")
print("📌 고정 모델 및 스케일러 저장 완료")