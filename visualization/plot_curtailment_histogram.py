import pandas as pd
import matplotlib.pyplot as plt

# ✅ 한글 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("D:/PythonProject/Curtailment_Predictor/data/final_dataset_for_LSTM.csv")

# 히스토그램 출력
plt.figure(figsize=(10, 5))
plt.hist(df['출력제한량'], bins=100, color='skyblue', edgecolor='black')
plt.title("출력제한량 분포 (히스토그램)")
plt.xlabel("출력제한량 (MWh)")
plt.ylabel("빈도 수")
plt.grid(True)
plt.tight_layout()
plt.show()
