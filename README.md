# 출력 제한 예측 기반 ESS 및 VPP 자율 대응 시스템

## 1. 프로젝트 개요
본 프로젝트는 제주지역 재생에너지 발전소의 출력 제한(curtailment) 문제를 해결하기 위해,  
기상·발전량·전력수요 데이터를 기반으로 출력 제한을 예측하고 이에 대응하는 ESS 충·방전 스케줄링 시스템을 구현하는 것을 목표로 한다.

## 2. 데이터 구성
- 발전량 데이터 (2017~2024, 한국전력거래소)
- 기상 데이터 (2022~2024, 기상청 4지점 평균, 풍향 제외)
- 전력수요 데이터 (제주지역 시간별 수요, 단위 보정 및 시간 보정 처리)

모든 데이터를 시간 단위(datetime 기준)로 통일하고 결측은 0으로 보완하였다.

## 3. 시스템 구조

```
기상/수요/발전량 정제 → 시계열 통합 → 상관분석 → 시계열 입력 생성 → LSTM 학습/예측 → ESS 운용 판단
```

## 4. 모델 설계
- 입력 시퀀스: 과거 24시간
- 출력: 다음 1시간 태양광 or 풍력 발전량
- 모델: LSTM (PyTorch 기반)
- 정규화: MinMaxScaler

## 5. 결과 요약
- 예측 정확도(MAE 기준): TBD
- 일사량과 태양광 발전량의 상관관계: 0.89
- 풍속과 풍력 발전량의 상관관계: 0.69
- 풍향 제거 결정: 벡터 평균의 불안정성으로 제외함

## 6. 폴더 구조

```
Curtailment_Predictor_Project/
├─ data/
│   ├─ raw/
│   ├─ processed/
│   └─ final/final_input_X.csv
├─ scripts/
│   ├─ preprocessing/
│   ├─ analysis/
│   ├─ modeling/
├─ results/
├─ reports/
└─ README.md
```
```mermaid
flowchart TD
    A[입력벡터 CSV 로드<br/>입력벡터_기록.csv] --> B[가장 최근 벡터 추출]
    B --> C[스케일러 로드<br/>scaler_fixed.pkl]
    C --> D[입력 벡터 스케일링]

    D --> E[모델 로드<br/>model_fixed.h5]
    E --> F[model.predict() 수행]

    F --> G[출력 제한량 계산<br/>round(prediction, 2)]
    G --> H[예측 결과 콘솔 출력]

    H --> I[결과 CSV 파일 생성/이어쓰기<br/>predicted_curtailments.csv]
```
```mermaid
graph TD
    P[main_predict.py] --> CSV[입력 데이터 CSV]
    P --> PKL[스케일러 scaler_fixed.pkl]
    P --> H5[모델 model_fixed.h5]
    P --> OUT[예측 결과 predicted_curtailments.csv]

    CSV --> P
    PKL --> P
    H5 --> P
    P --> OUT
```
```mermaid
sequenceDiagram
    participant U as User
    participant M as main_predict.py
    participant S as Scaler
    participant NN as LSTM Model
    participant R as Results CSV

    U->>M: 실행
    M->>M: 최신 입력 벡터 로드
    M->>S: scaler_fixed.pkl 불러오기
    S-->>M: scaled_input 반환
    M->>NN: model_fixed.h5 로드 후 예측
    NN-->>M: predicted_output
    M->>R: predicted_curtailments.csv 이어쓰기
```
```mermaid
flowchart LR
    KMA[기상청 API<br/>기상 데이터] --> API[api/ 스크립트<br/>데이터 수집]
    KPX[전력거래소 API<br/>발전·수요 데이터] --> API

    API --> PP[preprocessing/<br/>전처리 & 피처엔지니어링]
    PP --> DS[data/processed<br/>학습 입력·타깃]

    DS --> TRAIN[modeling/<br/>LSTM 학습(main_model.py)]
    TRAIN --> MODEL[저장된 모델<br/>model_fixed.h5]
    TRAIN --> SCALER[저장된 스케일러<br/>scaler_fixed.pkl]

    MODEL --> PRED[main_daily_run.py<br/>일일 예측]
    SCALER --> PRED
    API --> PRED

    PRED --> RES[results/<br/>predicted_curtailments.csv]
    RES --> SCHED[ESS·VPP 스케줄러<br/>(추가 구현)]
    SCHED --> HW[라즈베리파이·ESP32·부하<br/>실험 데모]
```
```mermaid
flowchart TD
    R1[data/raw/<br/>원시 CSV] --> C1[cleaning<br/>결측/단위 보정]
    C1 --> M1[merge<br/>기상+발전+수요 통합]
    M1 --> F1[feature engineering<br/>파생변수 생성]
    F1 --> W1[windowing<br/>과거 24h → X, 다음 1h → y]
    W1 --> S1[scaling<br/>MinMaxScaler]
    S1 --> T1[train/val split]

    T1 --> LSTM[LSTM 모델 학습<br/>modeling/]
    LSTM --> BEST[best model 저장<br/>model_fixed.h5]
    S1 --> SAVE_SCALER[scaler 저장<br/>scaler_fixed.pkl]

    BEST --> METRIC[results/<br/>성능 지표·그래프]
```
```mermaid
flowchart TD
    TS[Windows Task Scheduler<br/>23:00] --> BAT1[run_daily_vector.bat]
    BAT1 --> PY1[main_daily_run.py]

    PY1 --> A1[api/<br/>오늘 데이터 추가 수집]
    A1 --> P1[preprocessing/<br/>입력벡터 업데이트<br/>입력벡터_기록.csv]

    P1 --> PY2[main_model_predict.py<br/>(예측 스크립트)]
    PY2 --> LOADM[모델 & 스케일러 로드<br/>model_fixed.h5, scaler_fixed.pkl]
    LOADM --> PRED[model.predict()]

    PRED --> CSV1[results/<br/>predicted_curtailments.csv<br/>누적 저장]
    CSV1 --> ESS[ESS 스케줄링 로직<br/>(향후 적용)]
```
```mermaid
graph TD
    R[ess-vpp-project] --> A[analysis<br/>EDA·그래프]
    R --> B[api<br/>KMA/KPX 수집 코드]
    R --> C[data<br/>raw·processed·final]
    R --> D[modeling<br/>LSTM 모델 정의·학습]
    R --> E[preprocessing<br/>전처리 파이프라인]
    R --> F[results<br/>예측·성능 결과]
    R --> G[tools<br/>배치·유틸 스크립트]
    R --> H[utils<br/>공통 함수]
    R --> V[visualization<br/>보고서용 그림]

    R --> M1[main.py]
    R --> M2[main_model.py]
    R --> M3[main_daily_run.py]
    R --> B1[run_daily_vector.bat]
    R --> B2[run_model_predict.bat]
```
```mermaid
flowchart LR
    CSV[입력벡터_기록.csv<br/>마지막 행 로드] --> SCALE[scaler_fixed.pkl<br/>transform]
    SCALE --> MODEL[model_fixed.h5<br/>load_model]
    MODEL --> PRED[model.predict()<br/>출력제한량 예측]
    PRED --> OUT_CSV[predicted_curtailments.csv<br/>날짜·예측값 이어쓰기]
    PRED --> PRINT[콘솔 출력<br/>예측된 출력제한량]
```
