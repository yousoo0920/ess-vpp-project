# fetch_power_output.py

import requests
import pandas as pd
from datetime import datetime, timedelta

# ✅ 인증키 (Decoding된 상태)
API_KEY = "puFiIT2i7/qrypWX1grGK5uqjy2PI1T+M2xr4UoMAnB4/9a+ELL5zW9HnXnMo65Ovjx3aIOkGFozPqCMRFzC+g=="

# ✅ 어제 날짜 기준
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')


def fetch_power(date_str):
    url = "https://openapi.kospo.co.kr/openapi/sd/pwr/pwrPlant/generationStatus"

    params = {
        "serviceKey": API_KEY,
        "pageNo": "1",
        "numOfRows": "100",
        "resultType": "json",
        "searchDate": date_str,  # 📌 필수 파라미터
        "regionId": "1",  # 📌 필요 시 확인 (1: 전국, 2: 남부, ...?)
        "pwrKind": "1"  # 📌 발전종류코드 (1: 태양광, 2: 풍력 등)
    }

    response = requests.get(url, params=params)
    print("🔗 요청 URL:", response.url)

    if response.status_code == 200:
        data = response.json()
        try:
            items = data['response']['body']['items']
            if not items:
                print("❌ item 데이터 없음")
                return pd.DataFrame()
            df = pd.DataFrame(items)
            return df
        except:
            print("❌ JSON 파싱 실패 또는 items 없음")
            return pd.DataFrame()
    else:
        print("❌ 요청 실패:", response.status_code)
        return pd.DataFrame()


# ✅ 실행
if __name__ == "__main__":
    df_result = fetch_power(yesterday)
    if not df_result.empty:
        df_result.to_csv("D:/PythonProject/Curtailment_Predictor/data/power_generation_yesterday.csv", index=False)
        print("✅ 저장 완료")
    else:
        print("⚠️ 저장 생략: 빈 데이터")
