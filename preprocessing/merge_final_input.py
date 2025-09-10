import pandas as pd

def merge_final_inputs(
    weather_path: str,
    generation_path: str,
    demand_path: str,
    output_path: str
):
    # 1. 파일 불러오기
    weather = pd.read_csv(weather_path)
    generation = pd.read_csv(generation_path)
    demand = pd.read_csv(demand_path)

    # 2. datetime 형식 통일
    for df in [weather, generation, demand]:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 3. 병합 (왼→오 순서)
    merged = weather.merge(generation, on='datetime', how='left')
    merged = merged.merge(demand, on='datetime', how='left')

    # 4. NaN 처리 (병합 후 누락값은 0으로 대체)
    merged = merged.fillna(0)

    # 5. 소수점 반올림 (모든 수치 컬럼)
    for col in merged.columns:
        if col != 'datetime':
            merged[col] = pd.to_numeric(merged[col], errors='coerce').round(3)

    # 6. 저장
    merged.to_csv(output_path, index=False)
    print(f"✅ 병합 완료 → {output_path}")
    print(f"📊 총 행 수: {len(merged)}")

# ▶ 실행 예시
if __name__ == "__main__":
    merge_final_inputs(
        weather_path=r"D:\PythonProject\Curtailment_Predictor\data\weather_data.csv",
        generation_path=r"D:\PythonProject\Curtailment_Predictor\data\generation_data.csv",
        demand_path=r"D:\PythonProject\Curtailment_Predictor\data\demand_data.csv",
        output_path=r"D:\PythonProject\Curtailment_Predictor\data\final_input_X.csv"
    )
