import requests
import pandas as pd
from datetime import datetime, timedelta

SERVICE_KEY = "puFiIT2i7%2FqrypWX1grGK5uqjy2PI1T%2BM2xr4UoMAnB4%2F9a%2BELL5zW9HnXnMo65Ovjx3aIOkGFozPqCMRFzC%2Bg%3D%3D"
URL = (
    f"https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand"
    f"?serviceKey={SERVICE_KEY}"
    f"&pageNo=1&numOfRows=1000&dataType=JSON"
)

def fetch_demand_forecast(region_name="육지"):
    try:
        response = requests.get(URL)
        res_json = response.json()

        items = res_json['response']['body']['items']['item']
        df = pd.DataFrame(items)

        # 지역 필터링
        df = df[df['areaName'] == region_name]

        # 날짜 및 시간 필드 정리
        df['date'] = df['date'].astype(str).str.strip()
        df['hour'] = df['hour'].astype(str).str.strip()

        # hour == '24' 처리: 날짜 하루 더하고, 시간은 00으로 바꿈
        mask_24 = df['hour'] == '24'
        df.loc[mask_24, 'date'] = (
            pd.to_datetime(df.loc[mask_24, 'date'], format='%Y%m%d') + timedelta(days=1)
        ).dt.strftime('%Y%m%d')
        df.loc[mask_24, 'hour'] = '00'

        # 시간 문자열 자릿수 맞춤
        df['hour'] = df['hour'].str.zfill(2)

        # datetime 결합 및 변환
        df['datetime'] = pd.to_datetime(df['date'] + df['hour'], format='%Y%m%d%H')

        # 최종 정리
        df = df[['datetime', 'slfd']].rename(columns={'slfd': 'demand_forecast'})
        df['demand_forecast'] = df['demand_forecast'].astype(float)

        return df.sort_values('datetime').reset_index(drop=True)

    except Exception as e:
        print(f"❌ 파싱 실패: {e}")
        print("응답 내용:", response.text[:300])
        return None

if __name__ == "__main__":
    df = fetch_demand_forecast("육지")
    if df is not None:
        print(df.head())
        df.to_csv("demand_forecast.csv", index=False)
        print("✅ 저장 완료: demand_forecast.csv")
