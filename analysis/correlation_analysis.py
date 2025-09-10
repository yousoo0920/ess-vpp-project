import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 📂 CSV 파일 불러오기
df = pd.read_csv("D:/PythonProject/Curtailment_Predictor/data/final_input_X.csv")

# 🧹 datetime 제거하고 상관계수 계산
df_corr = df.drop(columns=["datetime"], errors="ignore")
corr_matrix = df_corr.corr(method="pearson")

# ✅ 한글 폰트 설정 (윈도우는 맑은 고딕, mac은 AppleGothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 📊 히트맵 그리기
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("상관관계 히트맵", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# 💾 저장 및 보여주기
plt.savefig("correlation_heatmap_korean.png", dpi=300)
plt.show()
