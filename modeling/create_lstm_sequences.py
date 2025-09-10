import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_lstm_sequences(df, feature_cols, sequence_length):
    X = []
    for i in range(len(df) - sequence_length):
        seq = df[feature_cols].iloc[i:i+sequence_length].values
        X.append(seq)
    return np.array(X)

if __name__ == "__main__":
    # ✅ 병합된 입력 CSV 경로 (수정된 절대 경로 or 상대 경로 사용)
    csv_path = "D:/PythonProject/Curtailment_Predictor/api/merged_input.csv"
    df = pd.read_csv(csv_path, parse_dates=['datetime'])

    # ✅ 사용할 feature 컬럼
    feature_cols = ['기온(°C)', '풍속(m/s)', '강수확률(%)', 'demand_forecast']

    # ✅ 결측치 제거 (예방용)
    df = df.dropna(subset=feature_cols)

    # ✅ 정규화 (MinMaxScaler 사용)
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # ✅ 시퀀스 길이 설정 (예: 24시간 시계열 입력)
    SEQ_LEN = 24

    # ✅ 입력 시퀀스 생성
    X = create_lstm_sequences(df, feature_cols, SEQ_LEN)

    # ✅ 저장
    np.save("X_lstm_input.npy", X)

    # ✅ 출력
    print(f"✅ 입력 시퀀스 shape: {X.shape} (samples, timesteps, features)")
    print("✅ 저장 완료: X_lstm_input.npy")
