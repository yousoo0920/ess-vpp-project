import requests
import pandas as pd
import xml.etree.ElementTree as ET

# 1. API 요청
url = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
params = {
    "serviceKey": "ENCODED_KEY",
    "pageNo": "1",
    "numOfRows": "1000",
    "dataType": "XML",
    "base_date": "20250605",
    "base_time": "0500",
    "nx": "60",  # 예시 좌표
    "ny": "127",
}
response = requests.get(url, params=params)

# 2. XML → DataFrame 파싱
root = ET.fromstring(response.text)
items = root.findall(".//item")

records = []
for item in items:
    record = {child.tag: child.text for child in item}
    records.append(record)

df = pd.DataFrame(records)

# 3. ✅ 여기 이 줄 추가!
df.to_csv("../data/weather/weather_short_forecast.csv", index=False, encoding="utf-8-sig")
print("✅ 기상 단기예보 저장 완료 → weather_short_forecast.csv")
