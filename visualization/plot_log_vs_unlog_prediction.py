import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. LSTM 데이터 불러오기
base_path = r"D:/PythonProject/Curtailment_Predictor/data/"
X = np.load(base_path + "X_lstm.npy")
Y = np.load(base_path + "Y_lstm.npy")

# 2. 날짜 불러오기 (전체 시계열에서 슬라이딩 윈도우 고려)
df = pd.read_csv(base_path + "final_input_X.csv")
dates_all = pd.to_datetime(df["datetime"])

# 🔧 슬라이딩 윈도우 offset 고려 (예: lookback=168)
lookback = X.shape[1]  # 윈도우 크기 자동 추출
dates_all = dates_all[lookback : lookback + len(Y)]  # Y와 정확히 맞춤

# ✅ 여기서 핵심: 뒤쪽 20%가 예측 대상임!
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
dates_train, dates_test = train_test_split(dates_all, test_size=0.2, shuffle=False)

# 4. 모델 불러오기
model = load_model(base_path + "lstm_model_output_curtailment.h5")

# 5. 예측
Y_pred_log = model.predict(X_test).flatten()
Y_true_log = Y_test

# 6. 로그 복원
Y_pred_unlog = np.expm1(Y_pred_log)
Y_true_unlog = np.expm1(Y_true_log)

# 7. 시각화
plt.figure(figsize=(14, 5))
plt.plot(dates_test, Y_true_unlog, label="실제 출력 제한량", color="blue")
plt.plot(dates_test, Y_pred_unlog, label="LSTM 예측 출력 제한량", color="orange")
plt.title("출력 제한량 예측 결과 비교: 실제값 vs LSTM 예측값 (Test Set)")
plt.xlabel("시간")
plt.ylabel("출력 제한량 (MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
