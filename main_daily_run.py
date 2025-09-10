# main_daily_run.py

from preprocessing.make_input_vector import make_input_vector_auto
from preprocessing.save_vector import save_vector_to_csv

if __name__ == "__main__":
    vec = make_input_vector_auto()
    if vec:
        print("✅ 벡터 생성됨:", vec)
        save_vector_to_csv(vec)
        print("✅ CSV에 저장 완료")
    else:
        print("❌ 입력벡터 생성 실패")
