# 📄 weather_lstm_preparer.py

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 🔐 인코딩된 인증키
service_key = 'puFiIT2i7%2FqrypWX1grGK5uqjy2PI1T%2BM2xr4UoMAnB4%2F9a%2BELL5zW9HnXnMo65Ovjx3aIOkGFozPqCMRFzC%2Bg%3D%3D'

# 📡 API 요청 URL
url = f"https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?serviceKey={service_key}"
params = {
    'pageNo': '1',
    'numOfRows': '1000',
    'dataType': 'JSON',
    'base_date': '20250604',
    'base_time': '1400',
    'nx': '59',
    'ny': '126'
}

# 📬 API 요청
response = requests.get(url, params=params)

# 🔍 JSON 파싱
try:
    data = response.json()
    items = data['response']['body']['items']['item']
    print("✅ JSON 파싱 성공")
except Exception as e:
    print("❌ JSON 파싱 실패:", e)
    print("🔎 응답 내용:\n", response.text[:500])
    exit()
print("🔗 요청 URL:", response.url)

# 🎯 필요한 항목만 필터링
target_categories = ['TMP', 'WSD', 'POP', 'SKY', 'PTY']
filtered = [item for item in items if item['category'] in target_categories]

# 📊 DataFrame 생성 및 정리
df = pd.DataFrame(filtered)
df['datetime'] = df['fcstDate'] + df['fcstTime']
df = df[['datetime', 'category', 'fcstValue']]
df_pivot = df.pivot(index='datetime', columns='category', values='fcstValue').reset_index()
df_pivot.columns.name = None

# 🔁 한글 컬럼명 적용
df_pivot = df_pivot.rename(columns={
    'TMP': '기온(°C)',
    'WSD': '풍속(m/s)',
    'POP': '강수확률(%)',
    'SKY': '하늘상태',
    'PTY': '강수형태',
    'datetime': '예보시각'
})

# ✅ 출력 확인
print("\n📊 변환된 최종 데이터 (상위 5개):\n")
print(df_pivot.head())

# ① datetime 변환 및 정렬
df_pivot['예보시각'] = pd.to_datetime(df_pivot['예보시각'], format='%Y%m%d%H%M')
df_pivot = df_pivot.sort_values(by='예보시각').reset_index(drop=True)

# ② 시각 제외한 입력값 추출
time_index = df_pivot['예보시각']
X_raw = df_pivot.drop(columns=['예보시각']).astype('float32')

# ③ 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# ④ 시퀀스 자르기 함수
def make_lstm_sequence(data, window_size=6):
    X_seq = []
    for i in range(len(data) - window_size + 1):
        X_seq.append(data[i:i+window_size])
    return np.array(X_seq)

# ⑤ 시퀀스 생성
window_size = 6
X_lstm = make_lstm_sequence(X_scaled, window_size)
time_seq = [time_index[i:i+window_size].tolist() for i in range(len(time_index) - window_size + 1)]

# ✅ 최종 출력 확인
print(f"\n📐 LSTM 입력 shape: {X_lstm.shape}")
print(f"🕒 첫 시퀀스 시각: {time_seq[0]}")
print(f"📈 첫 시퀀스 값:\n{X_lstm[0]}")

# ⑥ CSV 파일로 저장
df_pivot.to_csv("../data/weather/weather_short_forecast.csv", index=False, encoding='utf-8-sig')
print("✅ CSV 파일 저장 완료 → weather_short_forecast.csv")
