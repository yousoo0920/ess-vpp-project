import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# ✅ 한글 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic'    # Windows 사용자용
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 부호 깨짐 방지

# 1. 데이터 로드
base_path = r"D:/PythonProject/Curtailment_Predictor/data/"
X = np.load(base_path + "X_lstm.npy")
Y = np.load(base_path + "Y_lstm.npy")

# 2. 시간순 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# 3. 모델 불러오기
model = load_model(base_path + "lstm_model_output_curtailment.h5")

# 4. 예측 수행
Y_pred_log = model.predict(X_test).flatten()
Y_true_log = Y_test

# 5. 로그 역변환 (expm1 적용)
Y_pred = np.expm1(Y_pred_log)
Y_true = np.expm1(Y_true_log)

# 6. 시각화
plt.figure(figsize=(12, 5))
plt.plot(Y_true[:200], label='실제 출력제한량', linewidth=1.5)
plt.plot(Y_pred[:200], label='예측 출력제한량', linewidth=1.5)
plt.title("출력제한량 실제 vs 예측 (LSTM)")
plt.xlabel("시점 index")
plt.ylabel("출력제한량 (MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
