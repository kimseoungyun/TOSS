# 💳 2025 TOSS NEXT ML CHALLENGE: CTR Prediction

> **제4회 토스 NEXT 개발자 챌린지 - 광고 클릭 예측(CTR) 모델 개발** \> Transformer 기반의 Hybrid Tabular Model을 활용한 광고 클릭률 예측 솔루션입니다.

## 📌 Competition Overview

  * **대회명:** 토스 NEXT ML CHALLENGE : 광고 클릭 예측(CTR) 모델 개발
  * **주최/주관:** Viva Republica (Toss) / DACON
  * **링크:** [DACON Competition Page](https://dacon.io/competitions/official/236575/overview/description)
  * **목표:** 유저의 행동 로그와 광고 속성 데이터를 기반으로 광고 클릭 여부(`clicked`)를 예측하는 이진 분류(Binary Classification) 모델 개발.

## 📝 Dataset Description

총 119개의 컬럼으로 구성된 대규모 데이터셋을 활용했습니다.

| Feature Type | Columns | Description |
| :--- | :--- | :--- |
| **Target** | `clicked` | 광고 클릭 여부 (0/1) |
| **User Info** | `gender`, `age_group` | 유저 성별 및 연령대 |
| **Context** | `day_of_week`, `hour`, `inventory_id` | 노출 요일, 시간, 지면 ID |
| **Sequence** | `seq` | 유저의 과거 서버 로그 시퀀스 (Sequential Data) |
| **Ad Attributes** | `l_feat_*` | 광고 속성 정보 (14번: Ads set 등) |
| **Information** | `feat_a~e_*` | 정보 영역별 세부 피처 |
| **History** | `history_a_*` | 과거 인기도 및 이력 정보 |

## 🏗️ Model Architecture: Transformer for Tabular & Sequence

본 솔루션은 다양한 데이터 타입(범주형, 수치형, 시퀀스)을 효과적으로 융합하기 위해 **Transformer Encoder** 기반의 아키텍처를 설계했습니다. 각기 다른 성격의 데이터를 독립적인 모듈로 인코딩한 뒤, 이를 '토큰(Token)'화하여 Transformer가 피처 간의 상호작용(Interaction)을 학습하도록 구성했습니다.

### Model Diagram
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/259f44c5-bddd-41fa-a78a-5080302804b5" />
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/bb47eab1-ab08-4bfd-8a2d-b6a21496232e" />


### Key Components

1.  **Multi-Modal Inputs Processing:**

      * **Categorical:** `nn.Embedding`을 통해 고차원 벡터로 변환 후, Linear Projection을 통해 토큰화.
      * **Numerical:** MLP(Linear -\> ReLU -\> Linear)를 통과시켜 압축된 정보를 토큰화.
      * **Sequential (`seq`):** `nn.LSTM`을 활용하여 가변 길이의 유저 로그 시퀀스를 처리하고, 마지막 Hidden State를 추출하여 시퀀스 문맥 정보를 담은 토큰 생성.

2.  **Transformer Encoder:**

      * 범주형, 수치형, 시퀀스 모듈에서 생성된 임베딩들을 하나의 시퀀스(`[Cat_Token, Num_Token, Seq_Token]`)로 결합.
      * Self-Attention 메커니즘을 통해 데이터 타입 간의 복잡한 상관관계를 학습.

3.  **Prediction Head:**

      * 인코딩된 정보를 Flatten하여 MLP를 통과시켜 최종 클릭 확률 예측.

## 💻 Code Structure

### `TransformerTabularModel`

```python
# 핵심 모델 아키텍처 (Hybrid Transformer)
class TransformerTabularModel(nn.Module):
    def __init__(self, ...):
        # 1. Embedding & Projection
        self.cat_proj = nn.Linear(total_cat_emb_dim, 64)
        self.num_mlp = nn.Sequential(...) 
        
        # 2. Sequence Processing (LSTM)
        self.lstm = nn.LSTM(...) 
        
        # 3. Transformer Encoder (Feature Interaction)
        self.transformer = nn.TransformerEncoder(...)
        
        # 4. Final Classification
        self.final_mlp = nn.Sequential(...)

    def forward(self, cat_x, num_x, seq_x, seq_lengths):
        # ... (Forward propagation logic)
        tokens = torch.cat([cat_token, num_token, seq_token], dim=1)
        transformed = self.transformer(tokens)
        logits = self.final_mlp(pooled).squeeze(1)
        return logits
```

## ⚙️ Development Environment

  * **Language:** Python 3.x
  * **Deep Learning Framework:** PyTorch
  * **Libraries:**
      * `pandas`, `numpy`: 데이터 전처리
      * `scikit-learn`: 레이블 인코딩 및 스케일링
      * `torch`: 모델 구현 및 학습

## 🚀 How to Run

1.  **데이터 준비:**
      * DACON 대회 페이지에서 데이터를 다운로드하여 `./data` 경로에 위치시킵니다.
2.  **전처리 및 학습:**
      * 제공된 노트북 `TOSS_T2G_low.ipynb`를 실행합니다.
      * 노트북 내에는 데이터 로드, 전처리(결측치 처리, 인코딩), 모델 학습, 추론 과정이 포함되어 있습니다.

-----

### 📈 Future Works

  * **Feature Engineering:** `seq` 데이터 외에 시간 흐름에 따른 파생 변수 추가 생성.
  * **Ensemble:** Tree 기반 모델(XGBoost, CatBoost)과 Transformer 모델의 앙상블을 통한 성능 극대화.
  * **Hyperparameter Tuning:** Transformer layer 수 및 Head 수 최적화.
