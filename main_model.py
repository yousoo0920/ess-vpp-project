import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# ✅ 1. 최신 입력 벡터 불러오기
input_path = "D:/PythonProject/Curtailment_Predictor/data/입력벡터_기록.csv"
df = pd.read_csv(input_path)
latest_input = df.iloc[-1, 1:].values.reshape(1, -1)  # target_date 제외

# ✅ 2. 고정된 스케일러 로드
scaler = joblib.load("D:/PythonProject/Curtailment_Predictor/modeling/scaler_fixed.pkl")
scaled_input = scaler.transform(latest_input)

# ✅ 3. 고정된 모델 로드
model = load_model("D:/PythonProject/Curtailment_Predictor/modeling/model_fixed.h5")


# ✅ 4. 예측 수행
prediction = model.predict(scaled_input, verbose=0)
predicted_output = np.round(prediction[0][0], 2)

# ✅ 5. 콘솔 출력
print(f"📊 예측된 출력제한량: {predicted_output:.2f} MWh")

# ✅ 6. 결과 누적 저장
today = datetime.today().strftime("%Y-%m-%d")
result_path = "D:/PythonProject/Curtailment_Predictor/results/predicted_curtailments.csv"
new_row = pd.DataFrame([[today, predicted_output]], columns=["날짜", "예측 출력제한량(MWh)"])

# 기존 파일이 있으면 이어쓰기, 없으면 새로 생성
if os.path.exists(result_path):
    new_row.to_csv(result_path, mode='a', header=False, index=False)
else:
    new_row.to_csv(result_path, index=False)
