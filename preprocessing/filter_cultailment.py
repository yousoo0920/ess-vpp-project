import pandas as pd
import os

# ✅ 1. 절대 경로 설정
base_path = r"D:/PythonProject/Curtailment_Predictor/data/"

# ✅ 2. 실제 파일명에 맞게 정확히 입력 (공백 주의!)
files = [
    base_path + "한국전력거래소_월별 시간별 제주 태양광 풍력 제어량 및 제어 횟수_2017_2023.2.csv",    # 2022.1 ~ 2023.2
    base_path + "한국전력거래소_월별 시간별 제주 태양광 풍력 제어량 및 제어 횟수_2023.1 2023.12.csv",  # ✅ 공백 포함!
    base_path + "한국전력거래소_월별 시간별 제주 태양광 풍력 제어량 및 제어 횟수_2024.1_2024.3.csv",
    base_path + "한국전력거래소_월별 시간별 제주 태양광 풍력 제어량 및 제어 횟수_2024.4_2024.5.csv"
]

all_data = []

# ✅ 3. 디버깅용 파일 리스트 출력 (선택)
print("📂 현재 data 폴더 내 파일 목록:")
for f in os.listdir(base_path):
    print(" -", f)

# ✅ 4. 파일별 처리 루프
for file in files:
    print(f"\n📄 처리 중: {file}")

    df = pd.read_csv(file, encoding='cp949')  # 한글 인코딩

    # 날짜 컬럼 자동 인식
    if '기준일' in df.columns:
        date_col = '기준일'
    elif '일자' in df.columns:
        date_col = '일자'
    else:
        raise ValueError(f"{file}에서 날짜 컬럼을 찾을 수 없습니다.")

    # 시간 컬럼 (1시~24시 또는 1시간~24시간)
    hour_cols = [col for col in df.columns if (col.endswith('시') or col.endswith('시간')) and col[0].isdigit()]

    # melt 처리 (wide → long)
    df_melted = df.melt(id_vars=[date_col], value_vars=hour_cols,
                        var_name='시간', value_name='출력제한량')

    # datetime 생성
    df_melted[date_col] = pd.to_datetime(df_melted[date_col], errors='coerce')
    df_melted['시간숫자'] = df_melted['시간'].str.extract('(\d+)').astype(int)
    df_melted['datetime'] = df_melted[date_col] + pd.to_timedelta(df_melted['시간숫자'], unit='h')

    # 필요한 컬럼만 정리
    df_clean = df_melted[['datetime', '출력제한량']].copy()
    df_clean['출력제한량'] = pd.to_numeric(df_clean['출력제한량'], errors='coerce').fillna(0)

    # 날짜 필터링
    df_clean = df_clean[(df_clean['datetime'] >= '2022-01-01') & (df_clean['datetime'] <= '2024-05-31')]

    # 중복 제거 조건 적용
    if '2023.1 2023.12' in file:
        df_clean = df_clean[df_clean['datetime'] >= '2023-03-01']
    elif '2017_2023.2' in file:
        df_clean = df_clean[df_clean['datetime'] < '2023-03-01']

    all_data.append(df_clean)

# ✅ 5. 병합 및 정렬
final_df = pd.concat(all_data)
final_df = final_df.sort_values(by='datetime').reset_index(drop=True)

# ✅ 6. 중복 datetime 제거
final_df = final_df.drop_duplicates(subset='datetime', keep='first')

# ✅ 7. 출력제한여부 생성
final_df['출력제한여부'] = (final_df['출력제한량'] > 0).astype(int)

# ✅ 8. 저장
output_file = base_path + "Y_출력제한_정제완료.csv"
final_df.to_csv(output_file, index=False)
print(f"\n✅ 저장 완료: {output_file} (총 {len(final_df)}행)")
