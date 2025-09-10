import pandas as pd

def load_weather_data(path):
    df = pd.read_csv(path)

    # 🔹 출력제한량 계산 (컬럼이 있는 경우에만)
    if 'available_power' in df.columns and 'actual_output' in df.columns:
        df['curtailment_amount'] = df['available_power'] - df['actual_output']

    # 🔹 결측값 보간 및 제거
    df = df.interpolate().dropna()

    # 🔹 풍속 관련 파생변수 (wind_speed가 존재할 때만)
    if 'wind_speed' in df.columns:
        df = df[df['wind_speed'] >= 0]
        df['wind_speed_squared'] = df['wind_speed'] ** 2
        df['is_curtailment'] = df['wind_speed'].apply(lambda x: 1 if x >= 6.0 else 0)

    return df


# 🔸 이 아래 코드는 단독 실행할 때만 작동 (테스트용)
if __name__ == "__main__":
    df = load_weather_data("../data/weather_data.csv")
    print(df.head())
