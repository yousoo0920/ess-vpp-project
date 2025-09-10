import pandas as pd

# ✅ 1. 원본 CSV 불러오기
df = pd.read_csv("D:/PythonProject/Curtailment_Predictor/data/processed_dataset.csv")

# ✅ 2. 피처 엔지니어링
df["전일_출력제한량"] = df["출력제한량"].shift(1)
df["출력제한_변화율"] = (df["출력제한량"] - df["전일_출력제한량"]) / (df["전일_출력제한량"] + 1e-7)
df["전일_태양광"] = df["태양광발전량(MWh)"].shift(1)
df["전일_풍력"] = df["풍력발전량(MWh)"].shift(1)
df["출력비율"] = df["출력제한량"] / (df["태양광발전량(MWh)"] + df["풍력발전량(MWh)"] + 1e-7)

# ✅ 3. 결측치 제거
df.dropna(inplace=True)

# ✅ 4. 새 파일로 저장
df.to_csv("D:/PythonProject/Curtailment_Predictor/data/processed_dataset_with_engineering.csv", index=False)

print("✅ 저장 완료: processed_dataset_with_engineering.csv")
