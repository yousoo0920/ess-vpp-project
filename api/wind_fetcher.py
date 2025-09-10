import requests
import pandas as pd
import time

# ✅ 사용자 인증키 삽입 (이미 인코딩된 상태)
service_key = "puFiIT2i7%2FqrypWX1grGK5uqjy2PI1T%2BM2xr4UoMAnB4%2F9a%2BELL5zW9HnXnMo65Ovjx3aIOkGFozPqCMRFzC%2Bg%3D%3D"


def fetch_wind_generation(start_date, end_date, size=100):
    all_data = []
    page = 1

    while True:
        url = (
            f"https://apis.data.go.kr/B551893/wind-power-by-hour/list?"
            f"serviceKey={service_key}&startD={start_date}&endD={end_date}"
            f"&page={page}&size={size}"
        )
        print(f"\U0001f504 요청 URL: {url}")

        try:
            resp = requests.get(url, timeout=10)
            print("\U0001f4e1 응답 코드:", resp.status_code)
            if resp.status_code != 200:
                break

            if resp.text.startswith("<"):
                print("❌ JSON 파싱 오류: Expecting value: line 1 column 1 (char 0)")
                print("🔽 응답 원문:", resp.text)
                break

            json_data = resp.json()

            # OpenAPI 응답 내부 구조에서 'body' > 'content' 접근
            body = json_data.get("reponse", {}).get("body")
            if body is None:
                print("✅ 더 이상 데이터 없음.")
                break

            content = body.get("content")
            if not content:
                print("✅ 빈 페이지.")
                break

            all_data.extend(content)

            if len(content) < size:
                print("✅ 마지막 페이지 도달")
                break
            else:
                page += 1
                time.sleep(0.2)
        except Exception as e:
            print("❌ 예외 발생:", e)
            break

    if not all_data:
        print("⚠️ 데이터 없음: wind_generation_all_sites.csv 저장만 완료")
        pd.DataFrame().to_csv("wind_generation_all_sites.csv", index=False)
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # 날짜 및 시간 컬럼을 datetime으로 변환
    try:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["hour"].astype(str).str.zfill(2) + ":00:00",
            errors="coerce"
        )
    except Exception as e:
        print("⛔ datetime 파싱 실패:", e)

    df = df[["datetime", "generation", "site"]] if all(x in df.columns for x in ["datetime", "generation", "site"]) else df
    df.to_csv("wind_generation_all_sites.csv", index=False)
    print("✅ 저장 완료: wind_generation_all_sites.csv")
    return df


# ✅ 실행
if __name__ == "__main__":
    fetch_wind_generation("20240101", "20240107")
