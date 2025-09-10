# filename: make_input_vector.py

import pandas as pd
from datetime import datetime, timedelta
from api.power_generation_fetcher import get_solar_wind
import numpy as np
import time
from typing import Optional, Tuple

CURTAIL_CSV_PATH = "D:/PythonProject/Curtailment_Predictor/data/출력제한량.csv"

# ─────────────────────────────
# 내부 견고 래퍼: get_solar_wind 실패 방지
#  - 1차: 직통 호출 리트라이
#  - 2차: 근접 날짜(±1, ±2) 대체
#  - 3차: 완전 실패 시 (0.0, 0.0) 반환하고 계속 진행
# ─────────────────────────────
def _robust_get_solar_wind(day_str: str,
                           max_retry: int = 3,
                           nearby_days = (1, -1, 2, -2)) -> Tuple[float, float, str]:
    # 1) 직통 + 리트라이
    last_err = None
    for i in range(max_retry):
        try:
            data = get_solar_wind(day_str)
            solar = float(data.get("solar", 0.0))
            wind  = float(data.get("wind", 0.0))
            return solar, wind, "direct"
        except Exception as e:
            last_err = e
            # 지수백오프
            time.sleep(0.4 * (2 ** i))
    # 2) 근접 날짜 대체 (±1, ±2)
    base = datetime.strptime(day_str, "%Y%m%d")
    for off in nearby_days:
        alt = (base + timedelta(days=off)).strftime("%Y%m%d")
        for i in range(max_retry):
            try:
                data = get_solar_wind(alt)
                solar = float(data.get("solar", 0.0))
                wind  = float(data.get("wind", 0.0))
                return solar, wind, f"nearby:{off:+d}"
            except Exception:
                time.sleep(0.4 * (2 ** i))
    # 3) 완전 실패 → 0으로라도 계속
    print(f"[WARN] get_solar_wind 완전 실패 day={day_str}, last_err={last_err}")
    return 0.0, 0.0, "fallback:zero"

def _safe_read_curtail_df(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="cp949")
        # date를 datetime으로 확실히
        df['date'] = pd.to_datetime(df['date'].astype(str), format="%Y%m%d", errors="coerce")
        # 숫자 변환
        df['curtailment'] = pd.to_numeric(df['curtailment'], errors="coerce")
        return df
    except Exception as e:
        print(f"❌ 출력제한량 파일 불러오기 실패: {e}")
        return pd.DataFrame(columns=["date", "curtailment"])

def _get_curt_value(df: pd.DataFrame, yyyymmdd: str) -> float:
    if df.empty:
        return 0.0
    dt = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return 0.0
    series = df.loc[df['date'] == dt, 'curtailment']
    if series.empty:
        return 0.0
    val = series.values[0]
    try:
        return float(val) if pd.notna(val) else 0.0
    except Exception:
        return 0.0

def make_input_vector_for(target_date: str):
    """
    target_date(YYYYMMDD)의 입력 벡터를 생성.
    내부적으로 전일(D-1), 전전일(D-2)의 출력제한/발전량을 사용.
    """
    # 날짜 파생
    try:
        tgt_dt = datetime.strptime(target_date, "%Y%m%d")
    except Exception:
        print(f"❌ target_date 형식 오류: {target_date}")
        return None
    day_before = (tgt_dt - timedelta(days=1)).strftime("%Y%m%d")
    two_days_before = (tgt_dt - timedelta(days=2)).strftime("%Y%m%d")

    # 출력제한 CSV 읽기 (없거나 비어도 0으로 계속)
    df = _safe_read_curtail_df(CURTAIL_CSV_PATH)

    # 전일/전전일 출력제한
    c_today_val = _get_curt_value(df, day_before)       # D-1
    c_yesterday_val = _get_curt_value(df, two_days_before)  # D-2

    # 발전량(태양/풍력) — 견고 래퍼 사용
    solar, wind, src = _robust_get_solar_wind(day_before)
    if src != "direct":
        print(f"[WARN] 발전량 대체 사용 target={target_date}, D-1={day_before}, source={src}")

    # 파생 변수
    change_rate = (c_today_val - c_yesterday_val) / c_yesterday_val if c_yesterday_val not in (0, None) else 0.0
    denom = (solar + wind)
    output_ratio = (c_today_val / denom) if denom != 0 else 0.0

    input_vector = {
        "target_date": target_date,
        "전일_출력제한량": round(c_today_val, 4),
        "출력제한_변화율": round(change_rate, 4),
        "전일_태양광": round(solar, 2),
        "전일_풍력": round(wind, 2),
        "출력비율": round(output_ratio, 4)
    }
    return input_vector

def make_input_vector_auto():
    """기존 동작 유지: 오늘 날짜 기준으로 생성 (target_date=오늘)"""
    today = datetime.now().strftime("%Y%m%d")
    return make_input_vector_for(today)

# ✅ 실행 테스트 (저장까지는 save_vector.py에서 담당)
if __name__ == "__main__":
    result = make_input_vector_auto()
    if result:
        print("✅ 자동 생성된 입력 벡터:\n", result)
