import pandas as pd

# 1. 파일 경로 설정
base_path = r"D:/PythonProject/Curtailment_Predictor/data/"
x_file = base_path + "final_input_X.csv"
y_file = base_path + "Y_출력제한_정제완료.csv"

# 2. 파일 불러오기
df_x = pd.read_csv(x_file)
df_y = pd.read_csv(y_file)

# 3. datetime 파싱
df_x['datetime'] = pd.to_datetime(df_x['datetime'])
df_y['datetime'] = pd.to_datetime(df_y['datetime'])

# 4. datetime 기준 병합 (내부 조인)
df_merged = pd.merge(df_x, df_y, on='datetime', how='inner')

# 5. 결과 확인
print("✅ 병합 완료. 총 행 수:", len(df_merged))
print("📌 컬럼 목록:", df_merged.columns.tolist())

# 6. 저장
output_file = base_path + "final_dataset_for_LSTM.csv"
df_merged.to_csv(output_file, index=False)
print(f"📁 저장 완료: {output_file}")
