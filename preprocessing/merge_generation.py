import pandas as pd

def merge_generation_files(file1: str, file2: str, output_path: str):
    # 두 파일 불러오기
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # datetime 컬럼을 datetime 타입으로 변환
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    df2['datetime'] = pd.to_datetime(df2['datetime'])

    # 병합 및 중복 제거
    merged = pd.concat([df1, df2], ignore_index=True)
    merged = merged.drop_duplicates(subset='datetime').sort_values('datetime')

    # NaN → 0 으로 처리 (선택사항)
    merged.fillna(0, inplace=True)

    # 저장
    merged.to_csv(output_path, index=False)
    print(f"✅ 병합 완료 → {output_path}")

# 실행
if __name__ == "__main__":
    file1 = r"D:\PythonProject\Curtailment_Predictor\data\제주도_풍력태양광_시간별_202201_202311.csv"
    file2 = r"D:\PythonProject\Curtailment_Predictor\data\제주도_풍력태양광_시간별_202312_202412.csv"
    output_path = r"D:\PythonProject\Curtailment_Predictor\data\generation_data.csv"

    merge_generation_files(file1, file2, output_path)
