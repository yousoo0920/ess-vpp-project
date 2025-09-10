import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ 데이터 경로
data_path = "D:/PythonProject/Curtailment_Predictor/data/processed_dataset.csv"

# ✅ 데이터 로드
df = pd.read_csv(data_path)

# ✅ 타겟 컬럼: 출력제한여부
target_column = "출력제한여부"

# ✅ 사용할 feature 목록 (필요시 수정 가능)
feature_columns = [
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    '일조(hr)', '일사(MJ/m2)', '태양광발전량(MWh)',
    '풍력발전량(MWh)', '전력수요(MWh)'
]

# ✅ 입력/출력 분리
X = df[feature_columns].values
y = df[target_column].values

# ✅ 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ 모델 정의
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ 모델 학습
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# ✅ 예측 수행
y_pred_prob = model.predict(X_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# ✅ 예측 결과 저장
df['발생예측'] = y_pred

# ✅ 저장
df.to_csv("D:/PythonProject/Curtailment_Predictor/data/processed_dataset.csv", index=False)
print("✅ '발생예측' 컬럼 포함된 CSV 저장 완료!")
