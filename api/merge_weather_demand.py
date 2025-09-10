import pandas as pd

# 경로
weather_path = "D:/PythonProject/Curtailment_Predictor/data/weather/weather_short_forecast.csv"
demand_path = "demand_forecast.csv"

# ✅ 기상 데이터 로드
weather_df = pd.read_csv(weather_path)

# ✅ '예보시각'을 datetime으로 변환
weather_df['datetime'] = pd.to_datetime(weather_df['예보시각'])

# ✅ 수요예측 데이터 로드
demand_df = pd.read_csv(demand_path, parse_dates=['datetime'])

# ✅ 병합
merged_df = pd.merge(weather_df, demand_df, on="datetime", how="inner")

# ✅ 결과 확인 및 저장
print(merged_df.head())
merged_df.to_csv("merged_input.csv", index=False)
print("✅ 병합 완료: merged_input.csv")
