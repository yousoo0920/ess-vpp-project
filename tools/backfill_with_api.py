# tools/backfill_from_20250624.py
"""
누락 백필 스크립트 (2025-06-24 ~ 어제)
- 입력/출력 파일
  • 입력벡터_기록.csv : D:/PythonProject/Curtailment_Predictor/data/입력벡터_기록.csv
  • 출력제한량.csv     : D:/PythonProject/Curtailment_Predictor/data/출력제한량.csv
- 사용 API
  • 발전량: api.power_generation_fetcher.get_solar_wind(date_yyyymmdd_str)
  • (선택) 출력제한: api.curtailment_fetcher.get_curtailment(date_int_yyyymmdd)  # 없으면 자동 건너뜀
- 로직
  1) 출력제한량.csv 로드 (cp949)
  2) 누락된 target_date(=D)를 20250624~어제 범위에서 탐색
  3) 각 D에 대해 D-1, D-2의 curtailment로 파생 계산(변화율), D-1 발전량 API 호출
  4) preprocessing.save_vector.save_vector_to_csv()로 기록 (중복 방지)
  5) (선택) 출력제한량.csv에 과거 curtailment가 비어 있으면 API로 채움 (가능할 때만)
"""

import os
import sys
import time
import importlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd

# 고정 경로 (네 코드에 맞춤)
DATA_DIR = r"D:/PythonProject/Curtailment_Predictor/data"
FEATURES_CSV = os.path.join(DATA_DIR, "입력벡터_기록.csv")
CURTAIL_CSV  = os.path.join(DATA_DIR, "출력제한량.csv")

# 기본 범위: 로그 끊긴 다음날부터 어제
DEFAULT_START_INT = 20250624

# ──────────────────────────────
# 유틸
# ──────────────────────────────
def to_int(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d"))

def parse_int(y: int) -> datetime:
    return datetime.strptime(str(y), "%Y%m%d")

def daterange_int(start: int, end: int) -> List[int]:
    days = []
    d = parse_int(start)
    e = parse_int(end)
    while d <= e:
        days.append(to_int(d))
        d += timedelta(days=1)
    return days

def safe_read_csv(path: str, encoding="utf-8-sig") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding=encoding)

def safe_backup(path: str, encoding="utf-8-sig"):
    if not os.path.exists(path):
        return
    stem, ext = os.path.splitext(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{stem}.bak_{ts}{ext}"
    pd.read_csv(path, encoding=encoding).to_csv(backup, index=False, encoding=encoding)
    print(f"[백업] {path} -> {backup}")

def compute_change_rate(prev: float, prevprev: Optional[float]) -> float:
    if prevprev is None or prevprev == 0:
        return 0.0
    return (prev - prevprev) / prevprev

# ──────────────────────────────
# 기존 코드/함수 가져오기
# ──────────────────────────────
# 발전량 API: get_solar_wind(day_before_str_yyyymmdd)
from api.power_generation_fetcher import get_solar_wind
# 입력벡터 저장 유틸
from preprocessing.save_vector import save_vector_to_csv

# (선택) 출력제한 API가 있으면 자동 사용
def try_get_curtailment_from_api(date_int: int) -> Optional[float]:
    try:
        mod = importlib.import_module("api.curtailment_fetcher")
        fn  = getattr(mod, "get_curtailment", None)
        if callable(fn):
            return fn(date_int)  # float 또는 None
    except Exception:
        pass
    return None

# ──────────────────────────────
# 메인
# ──────────────────────────────
def backfill(start: Optional[int] = None, end: Optional[int] = None):
    # 1) 기간 결정
    if start is None:
        start = DEFAULT_START_INT
    if end is None:
        end = to_int(datetime.now() - timedelta(days=1))
    if start > end:
        raise ValueError(f"잘못된 기간: start({start}) > end({end})")
    print(f"[범위] {start} ~ {end}")

    # 2) CSV 로드
    #   - 출력제한량.csv는 기존 코드가 cp949로 저장해왔으므로 cp949로 로드
    curt_df = safe_read_csv(CURTAIL_CSV, encoding="cp949")
    if not curt_df.empty:
        # date를 int로 통일
        curt_df["date"] = curt_df["date"].astype(int)
        curt_df = curt_df.sort_values("date").reset_index(drop=True)
    else:
        curt_df = pd.DataFrame(columns=["date", "curtailment"])

    feat_df = safe_read_csv(FEATURES_CSV, encoding="utf-8-sig")
    if not feat_df.empty:
        feat_df["target_date"] = feat_df["target_date"].astype(int)
    else:
        feat_df = pd.DataFrame(columns=["target_date","전일_출력제한량","출력제한_변화율","전일_태양광","전일_풍력","출력비율"])

    # 3) 누락 날짜 계산
    full_days = set(daterange_int(start, end))
    existing_target_days = set(feat_df["target_date"].astype(int)) if not feat_df.empty else set()
    missing_targets = sorted(list(full_days - existing_target_days))
    print(f"[누락] 입력벡터 target_date: {len(missing_targets)}일")

    # 3-1) (선택) 출력제한 csv에 없는 날짜 채우기 (API 있으면)
    curt_days_existing = set(curt_df["date"].astype(int)) if not curt_df.empty else set()
    missing_curt_days = sorted(list(full_days - curt_days_existing))
    curt_added = 0
    if missing_curt_days:
        print(f"[참고] 출력제한량.csv 누락일 {len(missing_curt_days)}일 → API 시도")
        new_rows = []
        for d in missing_curt_days:
            val = try_get_curtailment_from_api(d)
            if val is not None:
                new_rows.append({"date": d, "curtailment": float(val)})
            # 과도한 호출 방지
            time.sleep(0.03)
        if new_rows:
            curt_df = pd.concat([curt_df, pd.DataFrame(new_rows)], ignore_index=True)
            curt_df = curt_df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            # 저장 (원본 인코딩 유지: cp949)
            safe_backup(CURTAIL_CSV, encoding="cp949")
            curt_df.to_csv(CURTAIL_CSV, index=False, encoding="cp949")
            curt_added = len(new_rows)
            print(f"[저장] 출력제한량.csv +{curt_added}일 (API 값)")
        else:
            print("[안내] 출력제한 API가 없거나 값이 없어 건너뜀")

    # 4) 빠른 조회를 위해 dict로 매핑
    curt_map = {int(r.date): float(r.curtailment) for r in curt_df.itertuples(index=False)}  # date -> curtailment

    # 5) 입력벡터 생성 및 저장
    added = 0
    for D in missing_targets:
        Dm1 = to_int(parse_int(D) - timedelta(days=1))
        Dm2 = to_int(parse_int(D) - timedelta(days=2))

        prev     = curt_map.get(Dm1, 0.0)
        prevprev = curt_map.get(Dm2, None)
        change_rate = compute_change_rate(prev, prevprev)

        # 발전량 API: D-1 문자열(YYYYMMDD)로 호출
        try:
            pv = get_solar_wind(str(Dm1))
            solar = float(pv.get("solar", 0.0))
            wind  = float(pv.get("wind", 0.0))
        except Exception as e:
            print(f"[경고] get_solar_wind 실패 D-1={Dm1}: {e}")
            solar, wind = 0.0, 0.0

        denom = (solar + wind)
        output_ratio = (prev / denom) if denom != 0 else 0.0

        vector = {
            "target_date": D,
            "전일_출력제한량": round(prev, 4),
            "출력제한_변화율": round(change_rate, 4),
            "전일_태양광": round(solar, 2),
            "전일_풍력": round(wind, 2),
            "출력비율": round(output_ratio, 4)
        }

        # 저장 (네 유틸 그대로 사용)
        save_vector_to_csv(vector, csv_path=FEATURES_CSV)
        added += 1

        # API rate limit 여유
        time.sleep(0.03)

    print(f"[결과] 입력벡터 추가: {added}일, 출력제한 추가: {curt_added}일")
    print("[완료] 백필 완료")

# ──────────────────────────────
# 엔트리
# ──────────────────────────────
if __name__ == "__main__":
    # 기본(20250624~어제)
    try:
        backfill()
    except KeyboardInterrupt:
        print("\n[중단] 사용자에 의해 중단")
        sys.exit(1)
