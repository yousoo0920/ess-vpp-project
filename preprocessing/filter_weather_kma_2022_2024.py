import pandas as pd
import os

def filter_weather_kma(input_paths: list, output_path: str):
    all_data = []

    for path in input_paths:
        try:
            df = pd.read_csv(path, encoding='cp949')
            print(f"📂 불러온 파일: {os.path.basename(path)}")

            df.columns = df.columns.str.strip()
            df = df[df['지점명'].isin(['제주', '서귀포', '성산', '고산'])]
            df['datetime'] = pd.to_datetime(df['일시'])

            # 풍향(16방위) 제외
            use_cols = [
                'datetime', '기온(°C)', '강수량(mm)', '풍속(m/s)',
                '습도(%)', '일조(hr)', '일사(MJ/m2)'
            ]
            df = df[use_cols]

            # 수치형 변환
            for col in use_cols[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            all_data.append(df)

        except Exception as e:
            print(f"⚠️ {path} 처리 오류: {e}")

    # 병합 및 평균
    full_df = pd.concat(all_data).groupby('datetime', as_index=False).mean()

    # 기준 시간 생성 및 병합
    full_range = pd.date_range("2022-01-01 00:00:00", "2024-12-31 23:00:00", freq='H')
    final_df = pd.DataFrame({'datetime': full_range})
    final_df = final_df.merge(full_df, on='datetime', how='left').fillna(0)

    # 소수점 반올림
    for col in final_df.columns:
        if col != 'datetime':
            final_df[col] = final_df[col].round(3)

    # 저장
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ 기상청 정제 완료 (풍향 제거) → {output_path}")
    print(f"📊 총 행 수: {len(final_df)}")


# ▶ 실행 예시
if __name__ == "__main__":
    filter_weather_kma(
        input_paths=[
            r"D:\PythonProject\Curtailment_Predictor\data\기상청 제주 시간별 데이터_2022.csv",
            r"D:\PythonProject\Curtailment_Predictor\data\기상청 제주 시간별 데이터_2023.csv",
            r"D:\PythonProject\Curtailment_Predictor\data\기상청 제주 시간별 데이터_2024.csv"
        ],
        output_path=r"D:\PythonProject\Curtailment_Predictor\data\weather_data.csv"
    )
