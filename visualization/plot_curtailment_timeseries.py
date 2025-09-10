import pandas as pd
import matplotlib.pyplot as plt

# ✅ 한글 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("D:/PythonProject/Curtailment_Predictor/data/final_dataset_for_LSTM.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')

plt.figure(figsize=(16, 5))
plt.plot(df['datetime'], df['출력제한량'], linewidth=0.7)
plt.title("전체 출력제한량 시계열 추이 (2022~2024)")
plt.xlabel("시간")
plt.ylabel("출력제한량 (MWh)")
plt.grid(True)
plt.tight_layout()
plt.show()
