import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_percentage_error

def run_regression(csv_path, feature_columns):
    # 데이터 불러오기
    df = pd.read_csv(csv_path)
    df_filtered = df[df["발생예측"] == 1]

    X = df_filtered[feature_columns].values
    y = df_filtered["출력제한량"].values

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 모델 생성
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mae")
    model.fit(X_train, y_train, epochs=30, verbose=1)

    # 예측
    y_pred = model.predict(X_val).flatten()

    return df_filtered, model, scaler, y_pred, y_val

# ✅ 실행부
csv_path = "D:/PythonProject/Curtailment_Predictor/data/processed_dataset_with_engineering.csv"
feature_columns = ["전일_출력제한량", "출력제한_변화율", "전일_태양광", "전일_풍력", "출력비율"]

df, model, scaler, y_pred, y_val = run_regression(csv_path, feature_columns)

# ✅ 모델 저장
model.save("D:/PythonProject/Curtailment_Predictor/data/regression_model_output_curtailment.h5")
print("✅ 모델 저장 완료")
