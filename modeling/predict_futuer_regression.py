# filename: predict_future_regression.py

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# ✅ 경로 설정
input_csv_path = "D:/PythonProject/Curtailment_Predictor/api/merged_input.csv"
model_path = "D:/PythonProject/Curtailment_Predictor/data/regression_model_output_curtailment.h5"
output_path = "D:/PythonProject/Curtailment_Predictor/results/future_prediction.csv"

# ✅ 과거와 동일한 feature 순서 유지
feature_columns = [
    "전일_출력제한량",
    "출력제한_변화율",
    "전일_태양광",
    "전일_풍력",
    "출력비율"
]

# ✅ 데이터 로딩
df = pd.read_csv(input_csv_path)

# ❗ 예측 대상이 되는 발생예측=1인 샘플만 선택 (과거와 같은 전처리 기준 유지)
if "발생예측" in df.columns:
    df = df[df["발생예측"] == 1]

# ✅ 입력값 추출 및 스케일링
X = df[feature_columns].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 과거에 사용한 scaler.pkl이 있다면 불러와야 하지만 현재는 새로 피팅함

# ✅ 모델 불러오기
model = load_model(model_path)

# ✅ 예측 수행
predictions = model.predict(X_scaled).flatten()

# ✅ 결과 저장
df_result = df.copy()
df_result["예측_출력제한량"] = predictions
df_result.to_csv(output_path, index=False)

print("✅ 미래 예측 완료 및 저장:", output_path)
