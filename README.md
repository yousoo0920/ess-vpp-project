# 모델 구조 상세 설명 및 시각화 자료

본 문서는 출력 제한 예측 모델의 **구조를 더 깊이 이해**하고  
**데이터 → 전처리 → 시계열 변환 → 모델 입력 → 모델 내부 흐름 → 예측 → ESS 스케줄링**  
전체 파이프라인을 한눈에 볼 수 있도록 정리한 확장판이다.

---

# 📌 1. End-to-End 전체 아키텍처 (가장 상위 구조)

```mermaid
flowchart LR
    A["기상청(KMA) API"] --> C["데이터 수집(api/)"]
    B["전력거래소(KPX) API"] --> C
    C --> D["전처리(preprocessing/)"]
    D --> E["파생 변수 생성(feature engineering)"]
    E --> F["시계열 윈도잉(Windowing)<br/>24시간 입력 → 1시간 예측"]
    F --> G["정규화(MinMaxScaler)<br/>scaler_fixed.pkl 저장"]
    G --> H["LSTM 학습(modeling/main_model.py)"]
    H --> I["최종 모델(h5) 저장<br/>model_fixed.h5"]

    I --> J["main_daily_run.py (일일 입력 벡터 생성)"]
    G --> J
    C --> J

    J --> K["main_model_predict.py<br/>출력 제한 예측"]
    K --> L["predicted_curtailments.csv 저장"]
    L --> M["ESS 충·방전 스케줄러(향후 추가)"]
```

---

# 📌 2. LSTM 모델 내부 구조 (자세한 내부 흐름도)

```mermaid
flowchart TD
    X["입력 X (batch, 24, features)"] --> L1["LSTM Layer 1<br/>(hidden=64, return_sequences=True)"]
    L1 --> DO1["Dropout(0.2)"]
    DO1 --> L2["LSTM Layer 2<br/>(hidden=32, return_sequences=False)"]
    L2 --> DO2["Dropout(0.2)"]
    DO2 --> D1["Dense(16) + ReLU"]
    D1 --> OUT["Dense(1)<br/>출력 제한량(MWh)"]
```

---

# 📌 3. 시계열 입력 생성(Windowing) 구조도  
24시간(24×N feature) → 다음 1시간 예측(y)

```mermaid
sequenceDiagram
    participant D as 데이터프레임
    participant W as Windowing
    participant X as X(Input)
    participant Y as y(Target)

    D->>W: 전체 시계열 데이터 전달
    W-->>X: 24시간 구간(슬라이딩 윈도우) 생성
    W-->>Y: X 다음 시점의 출력 제한량 생성
    X->>모델: (batch, 24, feature_size)
    Y->>모델: (batch, 1)
```

---

# 📌 4. 입력 특징(Features) 구조 (예시)

```mermaid
graph TD
    A["기상 데이터"] --> X["입력벡터"]
    B["풍속"] --> X
    C["일사량"] --> X
    D["기온"] --> X

    E["전력 수요"] --> X
    F["지역별 발전량"] --> X

    G["파생특성<br/>전일 대비 변화율"] --> X
    H["파생특성<br/>증감률(d/dt)"] --> X
    I["파생특성<br/>rolling mean / std"] --> X
```

---

# 📌 5. Feature Engineering 상세 구조

```mermaid
flowchart LR
    RAW["원시 데이터(raw)"] --> CLEAN["결측 처리 + 단위 통일"]
    CLEAN --> MERGE["기상 + 발전 + 수요 병합"]
    MERGE --> GEN1["전일 대비 변화량(d1)"]
    MERGE --> GEN2["변동성(rolling std)"]
    MERGE --> GEN3["증감률(gradient)"]
    GEN1 --> COMB["최종 Feature Matrix"]
    GEN2 --> COMB
    GEN3 --> COMB
```

---

# 📌 6. 모델 학습(Training) 전체 프로세스

```mermaid
flowchart TD
    A["데이터 로드"] --> B["전처리(정규화, NA 처리)"]
    B --> C["시계열 분할(windowing)"]
    C --> D["Train/Validation Split"]
    D --> E["LSTM 모델 구성"]
    E --> F["훈련(epoch 반복)"]
    F --> G["최적 모델 저장(model_fixed.h5)"]
    B --> H["Scaler 저장(scaler_fixed.pkl)"]
```

---

# 📌 7. 모델 입출력 텐서 흐름 (Tensor Flow)

```mermaid
flowchart LR
    A["입력 텐서 X<br/>(batch, 24, features)"] --> B["LSTM 1 (64 units)"]
    B --> C["LSTM 2 (32 units)"]
    C --> D["Dense 16"]
    D --> E["Dense 1"]
    E --> F["예측값(MWh)"]
```

---

# 📌 8. Attention-LSTM(옵션) 모델 구조도  
(만약 향후 개선 버전을 README에 기록하고 싶다면)

```mermaid
flowchart TD
    X["입력 X<br/>(batch,24,F)"] --> L["LSTM Encoder"]
    L --> AT["Attention Layer<br/>가중치 계산"]
    AT --> C["Context Vector"]
    C --> D["Dense Layer"]
    D --> Y["출력 제한량 예측"]
```

---

# 📌 9. 예측 스크립트( main_model_predict.py ) 처리 흐름

```mermaid
flowchart TD
    A["입력벡터_기록.csv 로드"] --> B["가장 최근 샘플 추출"]
    B --> C["scaler_fixed.pkl 로드<br/>→ transform"]
    C --> D["model_fixed.h5 로드"]
    D --> E["model.predict()"]
    E --> F["예측값 반올림"]
    F --> G["predicted_curtailments.csv 저장"]
    G --> H["ESS 스케줄링(향후 추가)"]
```

---

# 📌 10. 월간/주간 Error 분석 그래프 구조도 (EDA용)

### MAE 변화

```mermaid
graph LR
    A("2024-01") --> B("2024-02")
    B --> C("2024-03")
    C --> D("2024-04")
    D --> E("...")

    style A fill:#cfe3ff,stroke:#000
    style E fill:#cfe3ff,stroke:#000
```

### Error Distribution Concept

```mermaid
flowchart TD
    A["실제값(y_true)"] --> COMP["오차 = |y_true - y_pred|"]
    B["예측값(y_pred)"] --> COMP
    COMP --> HIST["오차 분포 히스토그램"]
    COMP --> PLOT["MAE / RMSE 시계열 플롯"]
```

(실제 그래프는 프로젝트의 `/visualization`에 저장)

---

# 📌 11. ESS 스케줄링 로직(추가 예정) 시각화

```mermaid
flowchart LR
    PRED["예측된 출력 제한량"] --> TH["임계치 판단"]
    TH -->|높음| CHG["ESS 충전 시작"]
    TH -->|낮음| DSG["ESS 방전 or 대기"]
    CHG --> U["VPP/부하 연동"]
    DSG --> U
```

---

# 📌 12. 향후 확장 가능한 모델 구조들 (문서용)

### CNN-LSTM 구조

```mermaid
flowchart TD
    X["입력 시퀀스"] --> CNN["1D CNN feature extractor"]
    CNN --> LSTM["LSTM Layer"]
    LSTM --> D["Dense Layers"]
    D --> Y["예측 결과"]
```

### Transformer Encoder 기반 구조

```mermaid
flowchart TD
    X["입력 (batch,24,F)"] --> T1["Multi-Head Attention"]
    T1 --> T2["Feed Forward Network"]
    T2 --> T3["Pooling"]
    T3 --> Y["출력 제한량 예측"]
```

---

# 📌 13. README에서 모델 구조 기술 예시 문장

> 본 모델은 LSTM 기반 시계열 예측 모델로,  
> 과거 24시간 입력(기상·수요·발전·파생특성)을 기반으로  
> 다음 1시간의 출력 제한량(MWh)을 예측한다.  
> 모델은 2개 LSTM 계층(64,32 units)과 Dropout, Dense 계층으로 구성되며  
> MinMaxScaler로 정규화된 입력 벡터를 사용한다.

---

# 📌 14. 전체 모델 구조 요약 그림 (최종 요약)

```mermaid
flowchart TD
    X["입력<br/>24h×Feature"] --> A["전처리·정규화"]
    A --> B["LSTM Encoder Layer 1"]
    B --> C["LSTM Encoder Layer 2"]
    C --> D["Dense Layer"]
    D --> Y["출력 제한 예측(MWh)"]
```

---

# ✔ 끝  
필요하다면 다음도 추가해줄 수 있음:

✅ 모델 학습 결과 시각 자료(예측 vs 실제 그래프)  
✅ 고급 Attention 가중치 시각화  
✅ ESS 동작 알고리즘 상세 flowchart  
✅ VPP 전체 계통 흐름도(3D 스타일 mermaid)  
✅ 프로젝트 논문 스타일 Introduction/Methodology/Experiment 섹션 정리  

원하는 스타일이 있으면 말해줘.
