# predict_with_trained_model.py

import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# ✅ 1. LSTM 모델 로드
model = load_model("your_trained_model.h5")  # ← 여기에 너의 학습 완료된 모델 파일 경로 넣기

# ✅ 2. 예측 입력 데이터 로드
X = np.load("X_lstm_input.npy")

# ✅ 3. 예측 수행
y_pred = model.predict(X, verbose=1)

# ✅ 4. 결과 출력 및 저장
print("✅ 예측 완료! 예측 shape:", y_pred.shape)

# 시계열 예측 결과를 시간정보 없이 저장 (추후 datetime 연결 가능)
df_pred = pd.DataFrame(y_pred, columns=["curtailment_prediction"])
df_pred.to_csv("curtailment_predicted.csv", index=False)
print("✅ 예측 결과 저장: curtailment_predicted.csv")
