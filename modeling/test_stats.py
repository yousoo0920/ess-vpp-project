import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ✅ 한글 폰트 설정
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# CSV 불러오기
df = pd.read_csv(r'D:\PythonProject\Curtailment_Predictor\data\regression_prediction.csv')

# 산점도 그리기
plt.scatter(df['출력제한량'], df['예측출력제한량'], alpha=0.5)
plt.xlabel('실제 출력제한량')
plt.ylabel('예측 출력제한량')
plt.title('출력제한량 예측 산점도')
plt.grid(True)
plt.show()
