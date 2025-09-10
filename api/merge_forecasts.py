# merge_forecasts.py

import pandas as pd

# 1. 파일에서 예보 데이터 불러오기
weather_df = pd.read_csv("../data/weather/weather_short_forecast.csv")  # 예시 파일명
power_df = pd.read_csv("../data/forecast_data.csv")  # power_demand_forecast_fetcher.py 결과

# 2. datetime 형식으로 컬럼 정리
weather_df["datetime"] = pd.to_datetime(weather_df["fcstDate"].astype(str) + weather_df["fcstTime"].astype(str).str.zfill(4), format="%Y%m%d%H%M")
power_df["datetime"] = pd.to_datetime(power_df["fcDate"].astype(str) + power_df["fcStime"].astype(str).str.zfill(2), format="%Y-%m-%d%H")

# 3. 병합
merged_df = pd.merge(weather_df, power_df, on="datetime", how="inner")

# 4. 저장
merged_df.to_csv("../data/combined_forecast.csv", index=False, encoding="utf-8-sig")

print("✅ 병합 완료: combined_forecast.csv 저장됨")
