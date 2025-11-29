# ⚡️ 출력 제한 예측 기반 ESS-VPP 자율 대응 시스템
> Autonomous ESS-VPP Response System based on Curtailment Prediction  
> Jeju Island Renewable Energy Curtailment Solution Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 1. 프로젝트 개요 (Executive Summary)

### 1.1 추진 배경
제주 지역 재생에너지 발전 비중 증가로 인해 계통 안정화를 위한 출력제어 발생이 급증하고 있다.  
본 프로젝트는 딥러닝 기반 예측과 ESS 자동 제어를 결합한 지능형 VPP 대응 시스템이다.

### 1.2 핵심 요약
- LSTM 기반 예측 모델 (MAE 1.98 MW)
- 출력제어 위험도 계산
- 위험도 기반 ESS 자동 충전/방전 운영

---

## 2. 시스템 아키텍처 (System Architecture)

```mermaid
flowchart TD

    subgraph Raw_Data_Layer ["Layer 1: Data Acquisition"]
        A["KPX 발전량 데이터"] --> D
        B["기상청 ASOS 데이터"] --> D
        C["제주 계통 수요 데이터"] --> D
    end

    subgraph Preprocessing_Layer ["Layer 2: Preprocessing"]
        D["시계열 데이터 통합"] --> E["결측치 보간"]
        E --> F["이상치 제거"]
        F --> G["Feature Engineering"]
        G --> H["MinMax Scaling"]
    end

    subgraph AI_Core_Layer ["Layer 3: Prediction Engine"]
        H --> I["Sliding Window (24h 입력)"]
        I --> J["LSTM Network"]
        J --> K["미래 1시간 발전량 예측"]
    end

    subgraph Control_Layer ["Layer 4: ESS Decision Logic"]
        K --> L{"Grid Capacity Check"}
        L --> M["Mode A: Curtailment Defense"]
        L --> N["Mode B: Economic Operation"]
        M --> O
        N --> O
        O["EMS Command Interface"]
    end
```

---

## 3. 데이터 및 모델링 상세

### 3.1 데이터셋 구성
| Feature Group | Variables | Description |
|---|---|---|
| Generation | PV_Amount, WT_Amount | 태양광·풍력 발전량 |
| Weather | Irradiance, WindSpeed | 일사량·풍속 |
| Grid | System_Load | 제주 전력 수요 |
| Time | Hour_Sin, Hour_Cos | 시간 주기성 인코딩 |

### 3.2 모델 구성

```mermaid
flowchart TD

    subgraph InputBlock ["Input Representation"]
        X["입력 X (24시간 × Feature)"]
    end

    subgraph EncoderBlock ["LSTM Encoder"]
        L1["LSTM 64 units"]
        D1["Dropout 0.2"]
        L2["LSTM 32 units"]
    end

    subgraph HeadBlock ["Prediction Head"]
        FC1["Dense 16 + ReLU"]
        FC2["Dense 1"]
    end

    subgraph OutputBlock ["Output"]
        Y["y_pred (MWh)"]
    end

    X --> L1 --> D1 --> L2 --> FC1 --> FC2 --> Y
```

### 3.3 예측 로직 시퀀스

```mermaid
sequenceDiagram
    participant S as Data Source
    participant P as Preprocessor
    participant M as LSTM Model
    participant C as Controller

    S->>P: Raw Data Fetch
    P->>P: Imputation and Scaling
    P->>M: Input Tensor [24h Window]
    M->>M: Forward Propagation
    M-->>C: Predicted Output
    C->>C: Grid Limit Check
    C->>S: ESS Command
```

---

## 4. 모델 성능 결과

### 4.1 정량적 지표
| Model | MAE | RMSE | R² | Notes |
|---|---:|---:|---:|---|
| ARIMA | 12.45 | 18.20 | 0.72 | Baseline |
| SVR | 8.32 | 11.05 | 0.81 | ML |
| Proposed LSTM | **1.98** | **2.85** | **0.98** | Best |

---

## 5. ESS 자율 운용 시뮬레이션

### 5.1 ESS Control Flow

```mermaid
flowchart TD
    A((Start)) --> B["데이터 수집"]
    B --> C["발전량 예측"]
    C --> D{"예측값 > 계통한계?"}

    D -->|YES| E["ESS 충전"]
    E --> X((End))

    D -->|NO| F{"전력가격 > 기준?"}
    F -->|YES| G["ESS 방전"]
    G --> X

    F -->|NO| H["대기 모드"]
    H --> X
```

---

## 6. 실행 방법

```bash
git clone https://github.com/yousoo0920/ess-vpp-project.git
cd ess-vpp-project
pip install -r requirements.txt
python main.py
```

---

## License
MIT License  
2025 ESS-VPP Project Team
