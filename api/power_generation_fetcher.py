# filename: power_generation_fetcher.py

import requests
import xml.etree.ElementTree as ET

def get_solar_wind(date: str):
    service_key = "puFiIT2i7%2FqrypWX1grGK5uqjy2PI1T%2BM2xr4UoMAnB4%2F9a%2BELL5zW9HnXnMo65Ovjx3aIOkGFozPqCMRFzC%2Bg%3D%3D"
    url = "https://openapi.kpx.or.kr/openapi/sumperfuel5m/getSumperfuel5m"
    full_url = f"{url}?serviceKey={service_key}&startTime={date}0000&endTime={date}2355"

    headers = {
        "Accept": "application/xml",
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(full_url, headers=headers)
    root = ET.fromstring(response.content)
    items = root.findall('.//item')

    solar_sum = 0
    wind_sum = 0

    for item in items:
        try:
            solar = float(item.findtext("fuelPwr8") or 0)
            wind = float(item.findtext("fuelPwr9") or 0)
            solar_sum += solar
            wind_sum += wind
        except:
            continue

    return {
        "date": date,
        "solar": round(solar_sum, 2),
        "wind": round(wind_sum, 2)
    }

# ✅ 테스트 실행
if __name__ == "__main__":
    from datetime import datetime, timedelta

    # 어제 날짜로 테스트
    target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    result = get_solar_wind(target_date)
    print("📊 발전량 요약:", result)
