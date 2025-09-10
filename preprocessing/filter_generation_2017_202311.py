import pandas as pd
import os

def filter_generation_old(input_paths: list, output_path: str):
    all_data = []

    for path in input_paths:
        try:
            df = pd.read_csv(path, encoding='cp949')
            print(f"📂 불러온 파일: {path}")

            # 컬럼 이름 공백 제거
            df.columns = df.columns.str.strip()

            # '지역명' 또는 '지역' 중 실제 존재하는 컬럼 찾기
            region_col = next((col for col in ['지역명', '지역'] if col in df.columns), None)
            if not region_col:
                raise ValueError("❌ '지역명' 또는 '지역' 컬럼이 없음")

            # 제주/제주도 데이터만 필터링
            df = df[df[region_col].astype(str).str.strip().isin(['제주', '제주도'])]

            # 발전량 컬럼 추출 함수
            def match_column(possibles):
                for col in df.columns:
                    col_normalized = col.replace(" ", "").lower()
                    for p in possibles:
                        if p.replace(" ", "").lower() in col_normalized:
                            return col
                return None

            # 발전량 컬럼 자동 인식
            solar_col = match_column(['태양광발전량(MWh)', '태양광 발전량', '태양광'])
            wind_col = match_column(['풍력발전량(MWh)', '풍력 발전량', '풍력'])

            if not solar_col or not wind_col:
                raise ValueError("❌ 발전량 컬럼 인식 실패")

            # datetime 생성
            df['datetime'] = pd.to_datetime(df['거래일자']) + pd.to_timedelta(df['거래시간'] - 1, unit='h')
            df = df[['datetime', solar_col, wind_col]].copy()
            df.columns = ['datetime', '태양광발전량(MWh)', '풍력발전량(MWh)']

            # 수치형 변환
            for col in ['태양광발전량(MWh)', '풍력발전량(MWh)']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            all_data.append(df)

        except Exception as e:
            print(f"⚠️ {os.path.basename(path)} 처리 오류: {e}")

    if not all_data:
        print("❌ 처리할 유효한 파일이 없습니다.")
        return

    # 하나로 합치고 중복 제거
    full_df = pd.concat(all_data).drop_duplicates('datetime')
    full_df = full_df.groupby('datetime', as_index=False).mean()

    # 기준 시간 생성: 2022.01.01 ~ 2023.11.30 23:00
    full_range = pd.date_range(start="2022-01-01 00:00:00", end="2023-11-30 23:00:00", freq='H')
    final_df = pd.DataFrame({'datetime': full_range})

    # 병합 후 누락값은 0으로 채움
    final_df = final_df.merge(full_df, on='datetime', how='left').fillna(0)

    # 저장
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ 발전량(2017~2023.11) 정제 완료 → {output_path}")
    print(f"📊 생성된 행 수: {len(final_df)}")

# ▶ 실행 예시
if __name__ == "__main__":
    filter_generation_old(
        input_paths=[
            r"D:\PythonProject\Curtailment_Predictor\data\한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2017_2023.2.csv",
            r"D:\PythonProject\Curtailment_Predictor\data\한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2023.3_2023.5.csv",
            r"D:\PythonProject\Curtailment_Predictor\data\한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2023.6_2023.8.csv",
            r"D:\PythonProject\Curtailment_Predictor\data\한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2023.9_2023.11.csv"
        ],
        output_path=r"D:\PythonProject\Curtailment_Predictor\data\제주도_풍력태양광_시간별_202201_202311.csv"
    )
