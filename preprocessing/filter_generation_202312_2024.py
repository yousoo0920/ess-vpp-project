import pandas as pd

def filter_generation_old_format(input_paths: list, output_path: str):
    all_data = []

    for path in input_paths:
        try:
            df = pd.read_csv(path, encoding='cp949')
            print(f"📂 불러온 파일: {path}")
            df.columns = df.columns.str.strip()

            # 지역 컬럼 자동 탐색
            region_col = next((col for col in ['지역명', '지역'] if col in df.columns), None)
            if region_col is None:
                raise ValueError("⚠️ '지역명' 또는 '지역' 컬럼이 없습니다.")

            # 발전량 컬럼 자동 탐색
            value_col = next((col for col in ['전력거래량(MWh)', '발전량(MWh)'] if col in df.columns), None)
            if value_col is None:
                raise ValueError("⚠️ '전력거래량(MWh)' 또는 '발전량(MWh)' 컬럼이 없습니다.")

            # 제주 지역 필터링
            df = df[df[region_col].astype(str).str.strip().isin(['제주', '제주도'])].copy()

            # 시간 컬럼 생성
            df['datetime'] = pd.to_datetime(df['거래일자']) + pd.to_timedelta(df['거래시간'] - 1, unit='h')

            # 태양광/풍력 분리
            solar_df = df[df['연료원'].str.contains('태양광')][['datetime', value_col]].copy()
            wind_df = df[df['연료원'].str.contains('풍력')][['datetime', value_col]].copy()

            solar_df.rename(columns={value_col: '태양광발전량(MWh)'}, inplace=True)
            wind_df.rename(columns={value_col: '풍력발전량(MWh)'}, inplace=True)

            # 병합
            merged_df = pd.merge(solar_df, wind_df, on='datetime', how='outer').sort_values('datetime')
            all_data.append(merged_df)

        except FileNotFoundError:
            print(f"❌ 파일 없음: {path}")
        except Exception as e:
            print(f"⚠️ 에러 발생 ({path}): {e}")

    if all_data:
        full_df = pd.concat(all_data).sort_values('datetime')
        full_df.to_csv(output_path, index=False)
        print(f"\n✅ 발전량 정제 완료 → {output_path}")
    else:
        print("⚠️ 불러올 데이터가 없습니다.")

# ✅ 실행 블록
if __name__ == "__main__":
    input_paths = [
        r"D:\PythonProject\Curtailment_Predictor\data\한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2023.12.csv",
        r"D:\PythonProject\Curtailment_Predictor\data\한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2024.csv"
    ]

    output_path = r"D:\PythonProject\Curtailment_Predictor\data\제주도_풍력태양광_시간별_202312_202412.csv"
    filter_generation_old_format(input_paths, output_path)
