import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os

# 1. 파일 경로 설정
base_path = r"D:/PythonProject/Curtailment_Predictor/data/"
input_file = base_path + "final_dataset_for_LSTM.csv"

# 2. 데이터 불러오기 및 정렬
df = pd.read_csv(input_file)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime').reset_index(drop=True)

# 3. 로그 변환 (출력제한량만)
df['출력제한량'] = np.log1p(df['출력제한량'])  # log(1 + x)

# 4. 컬럼 분리
target_col = '출력제한량'
exclude_cols = ['datetime', '출력제한량', '출력제한여부']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# 5. Robust 정규화
scaler = RobustScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

# 6. 시계열 입력 구성 (24시간 → 1시간 예측)
sequence_length = 24
X, Y = [], []

for i in range(len(df_scaled) - sequence_length):
    x_seq = df_scaled[feature_cols].iloc[i:i+sequence_length].values
    y_val = df_scaled[target_col].iloc[i + sequence_length]
    X.append(x_seq)
    Y.append(y_val)

X = np.array(X)
Y = np.array(Y)

# 7. 저장
np.save(base_path + "X_lstm.npy", X)
np.save(base_path + "Y_lstm.npy", Y)

print("✅ 로그 + Robust 정규화 포함 LSTM 입력 구성 완료")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
