# preprocessing/save_vector.py

import os
import pandas as pd

def save_vector_to_csv(vector: dict, csv_path="D:/PythonProject/Curtailment_Predictor/data/입력벡터_기록.csv"):
    df = pd.DataFrame([vector])

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
