import pandas as pd
import matplotlib.pyplot as plt

# ✅ 한글 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 원본 데이터 로드 (cp949 인코딩)
df_raw = pd.read_csv(
    "D:/PythonProject/Curtailment_Predictor/data/한국전력거래소_월별 시간별 제주 태양광 풍력 제어량 및 제어 횟수_2023.1 2023.12.csv",
    encoding='cp949'
)

# 2. 기준일을 기준으로 melt
df_melted = df_raw.melt(id_vars=['기준일'], var_name='시간', value_name='출력제한량')

# 3. datetime 생성
df_melted['datetime'] = pd.to_datetime(
    df_melted['기준일'] + ' ' + df_melted['시간'].str.replace('시', '') + ':00',
    errors='coerce'
)
df_melted = df_melted.dropna(subset=['datetime'])

# 4. 정제 후 데이터 불러오기
df_clean = pd.read_csv("D:/PythonProject/Curtailment_Predictor/data/final_dataset_for_LSTM.csv")
df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])

# 5. 시각화
plt.figure(figsize=(14, 6))
plt.plot(df_melted['datetime'], df_melted['출력제한량'], label='정제 전', alpha=0.4)
plt.plot(df_clean['datetime'], df_clean['출력제한량'], label='정제 후', alpha=0.8)
plt.title("출력제한량 데이터 정제 전후 비교 (2023년)")
plt.xlabel("시간")
plt.ylabel("출력제한량 (MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
