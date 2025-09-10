import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# ✅ 한글 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic'    # Windows 전용
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 부호 깨짐 방지

# 1. 경로 설정 및 데이터 불러오기
base_path = r"D:/PythonProject/Curtailment_Predictor/data/"
X = np.load(base_path + "X_lstm.npy")
Y = np.load(base_path + "Y_lstm.npy")

# 2. 시간 순서 유지한 Train/Test 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# 3. 모델 로드
model = load_model(base_path + "lstm_model_output_curtailment.h5")

# 4. 예측 수행 (log → 역변환)
Y_pred_log = model.predict(X_test)
Y_pred = np.expm1(Y_pred_log)
Y_test_true = np.expm1(Y_test)

# 5. 예측 오차 계산
errors = Y_test_true - Y_pred  # shape: (N, 1) → flatten 필요

# 6. 히스토그램 시각화
plt.figure(figsize=(10, 5))
plt.hist(errors.ravel(), bins=100, color='salmon', edgecolor='black')  # ✅ flatten
plt.title("예측 오차 분포 (실제값 - 예측값)")
plt.xlabel("오차 (MWh)")
plt.ylabel("빈도 수")
plt.grid(True)
plt.tight_layout()
plt.show()
