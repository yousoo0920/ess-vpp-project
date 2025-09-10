import pandas as pd

def filter_demand_data(input_paths: list, output_path: str):
    all_data = []

    for path in input_paths:
        try:
            df = pd.read_csv(path, encoding='cp949')
            print(f"📂 불러온 파일: {path}")

            # 컬럼 정리
            df.columns = df.columns.str.strip()

            # '거래일자' → datetime 변환
            df['거래일자'] = pd.to_datetime(df['거래일자'])

            # 시간별 수요 데이터(1시~24시)를 long-format으로 변환
            hourly = df.melt(id_vars=['거래일자'], var_name='시간', value_name='전력수요(MWh)')

            # '시간' 컬럼 처리: '1시' → 1, ..., '24시' → 24
            hourly['시간'] = hourly['시간'].str.replace('시', '').astype(int)

            # 정확한 시간 계산: +1시간 시프트 (ex: 1시 → 01:00:00 → 1:00~2:00 구간으로 해석)
            hourly['datetime'] = hourly['거래일자'] + pd.to_timedelta(hourly['시간'], unit='h')

            # 필요 컬럼만 정리
            result = hourly[['datetime', '전력수요(MWh)']]

            # 단위 통일: kWh → MWh 변환 필요 시 (지금은 이미 MWh로 되어있다면 생략 가능)
            # result['전력수요(MWh)'] = result['전력수요(MWh)'] / 1000

            all_data.append(result)

        except Exception as e:
            print(f"⚠️ 오류 발생 ({path}): {e}")

    # 병합 및 중복 제거
    full_df = pd.concat(all_data).drop_duplicates('datetime')

    # 기준 시간대 생성
    full_range = pd.date_range("2022-01-01 00:00:00", "2024-12-31 23:00:00", freq='H')
    base = pd.DataFrame({'datetime': full_range})

    # 병합 및 누락값 처리
    final_df = base.merge(full_df, on='datetime', how='left').fillna(0)
    final_df['전력수요(MWh)'] = final_df['전력수요(MWh)'].round(3)

    # 저장
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ 전력수요 정제 완료 → {output_path}")
    print(f"📊 총 행 수: {len(final_df)}")


# ▶ 실행 예시
if __name__ == "__main__":
    filter_demand_data(
        input_paths=[
            r"D:\PythonProject\Curtailment_Predictor\data\시간별 제주전력수요_2017_2023.2.csv",
            r"D:\PythonProject\Curtailment_Predictor\data\시간별 제주전력수요_2023.csv",
            r"D:\PythonProject\Curtailment_Predictor\data\시간별 제주전력수요_2024.csv"
        ],
        output_path=r"D:\PythonProject\Curtailment_Predictor\data\demand_data.csv"
    )
