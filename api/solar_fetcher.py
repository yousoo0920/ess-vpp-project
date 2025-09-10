import datetime
import os
import pandas as pd
from zeep import Client
from zeep.transports import Transport
from zeep.exceptions import Fault
import requests

# 🔑 인증키 (Decoding된 형태)
SERVICE_KEY = "puFiIT2i7/qrypWX1grGK5uqjy2PI1T+M2xr4UoMAnB4/9a+ELL5zW9HnXnMo65Ovjx3aIOkGFozPqCMRFzC+g=="

# 📡 WSDL URL (테스트 포함)
WSDL_URL = "https://apis.data.go.kr/B551184/openapi/service/SolarPvService?wsdl"

# ✅ 위치 및 시간 정보
LAT = 33.202807954
LON = 126.26336105
today = datetime.datetime.now().strftime("%Y%m%d")
now_hour = datetime.datetime.now().strftime("%H") + "00"


def fetch_solar():
    session = requests.Session()
    session.verify = True  # 필요시 False도 가능
    transport = Transport(session=session, timeout=10)

    client = Client(wsdl=WSDL_URL, transport=transport)

    # SOAP 함수명은 WSDL 내 확인 필요 (가정: getSolarPvPredict)
    service = client.bind('SolarPvService', 'SolarPvServiceSoap')

    try:
        result = service.getSolarPvPredict(
            serviceKey=SERVICE_KEY,
            date=today,
            time=now_hour,
            lat=LAT,
            lon=LON
        )
    except Fault as err:
        print("❌ SOAP 호출 오류:", err)
        return None

    return result


if __name__ == "__main__":
    data = fetch_solar()
    if not data:
        exit(1)

    # 반환 결과는 XML 구조 (zeep가 객체로 변환)
    try:
        items = data['body']['items']['item']
    except Exception:
        items = data

    df = pd.DataFrame([items] if isinstance(items, dict) else items)

    os.makedirs("data", exist_ok=True)
    fname = f"data/solar_soap_{today}_{now_hour}.csv"
    df.to_csv(fname, index=False, encoding='utf-8-sig')
    print("✅ SOAP 기반 저장 완료:", fname)
