# Toss 광고 클릭 최적화 프로젝트 (강화학습 수업)
Multi-Armed Bandit 및 Contextual Bandit 기반 토스 앱 광고 추천 정책 설계와 성능 평가
강화학습(Multi-Armed Bandit, Contextual Bandit)을 이용하여 Toss 앱 광고의 CTR(Click-Through Rate)을 최적화한 프로젝트입니다.  
이 저장소는 수업 프로젝트 제출을 위해 작성되었으며, 코드와 학습된 모델, 보고서(PPT&PDF)를 포함합니다.

---

## 1. 팀 정보

- 과목: 강화학습의 기초
- 팀장 & 팀원: A71030 배정언

---



## 2. 파일 구성

이 저장소의 루트에는 아래 파일들이 있습니다.

    .
    ├── Toss_Data preprocessing_EDA.ipynb
    ├── RL_TOSS_MAB_Stateless.ipynb
    ├── RL_TOSS_MAB_Contextual Bandit.ipynb
    ├── Toss_preprocessed_FF.csv
    └── README.md

- `Toss_Data preprocessing_EDA.ipynb`  
  - 원본 데이터(`sampled_500k_stratified.csv`)를 불러와 EDA, 피처 엔지니어링, 필터링 등을 수행하고  
    최종 학습용 데이터인 `Toss_preprocessed_FF.csv`를 생성하는 노트북입니다.
- `RL_TOSS_MAB_Stateless.ipynb`  
  - 전처리된 데이터(`Toss_preprocessed_FF.csv`)를 기반으로  
    ε-Greedy, UCB1, Thompson Sampling, Softmax 등 **Stateless Multi-Armed Bandit** 알고리즘을 구현하고  
    각 알고리즘의 CTR, Regret, 팔 선택 분포를 비교합니다.
- `RL_TOSS_MAB_Contextual Bandit.ipynb`  
  - 동일한 전처리 데이터를 사용하여  
    요일/시간/최근 노출 이력 등 컨텍스트를 반영한 **Contextual Bandit (LinUCB)** 알고리즘을 구현하고  
    Baseline 및 Stateless MAB와 성능을 비교합니다.
- `Toss_preprocessed_FF.csv`  
  - 실제 밴딧 실험에 사용되는 최종 전처리 데이터셋입니다.  
  - 용량이 약 20MB 수준으로, GitHub 저장소에 포함되어 있습니다.

> 원본 데이터 `sampled_500k_stratified.csv`는 용량 문제로 GitHub에 포함되어 있지 않고,  
> 아래 3.1절의 Google Drive 링크로 제공합니다.

---

## 3. 데이터

### 3.1. 원본 데이터 (대용량, Google Drive 제공)

- 파일명: `sampled_500k_stratified.csv`
- 설명: Toss 광고 로그에서 **층화 샘플링(stratified sampling)**으로 추출한 약 50만 행 규모의 데이터셋입니다.
- 용량이 커서 GitHub에 직접 업로드하기 어려워, Google Drive 링크로 제공합니다.

**원본 데이터 다운로드 링크**

👉 [sampled_500k_stratified.csv (Google Drive)](https://drive.google.com/file/d/1tYpVOicfixHA_8lDwkUbsi5IeR6db-kK/view?usp=sharing)

Colab에서 이 파일을 사용하려면,

1. 위 링크에서 파일을 다운로드하여 Colab 작업 디렉터리에 업로드하거나  
2. Google Drive를 마운트한 후 해당 경로를 지정해서 사용합니다.

예시:

    import pandas as pd

    # (예시) Colab 노트북과 같은 위치에 csv를 둔 경우
    df_raw = pd.read_csv("sampled_500k_stratified.csv")

### 3.2. 전처리 데이터 (`Toss_preprocessed_FF.csv`)

- `Toss_Data preprocessing_EDA.ipynb`를 통해 생성된 최종 학습용 데이터셋입니다.
- GitHub 저장소에 포함되어 있으며, 밴딧 실험 노트북에서 바로 사용할 수 있습니다.

예시:

    import pandas as pd

    df = pd.read_csv("Toss_preprocessed_FF.csv")

---

## 4. 개발 환경

- 실행 환경: Google Colab (Python 3.12)
- 주요 라이브러리
  - numpy  
  - pandas  
  - scikit-learn  
  - matplotlib  
  - (필요 시) seaborn, tqdm 등  

로컬에서 실행할 경우, 위 패키지들을 `pip install 패키지명`으로 설치한 뒤  
Jupyter Notebook에서 각 `.ipynb` 파일을 열어 실행하면 됩니다.

---

## 5. 노트북 실행 방법

### 5.1. 전처리 & EDA 노트북 (선택)

> `Toss_preprocessed_FF.csv` 파일이 이미 제공되므로, 아래 과정은 **선택 사항**입니다.  
> 전처리 과정을 다시 수행하거나 EDA 과정을 확인하고 싶을 때만 실행하면 됩니다.

1. `Toss_Data preprocessing_EDA.ipynb` 파일을 열어 상단부터 순서대로 셀을 실행합니다.  
2. 원본 데이터 경로를 실제 위치에 맞게 설정합니다. 예시는 다음과 같습니다.  

       import pandas as pd
       df_raw = pd.read_csv("sampled_500k_stratified.csv")  # 또는 드라이브/다른 경로

3. 노트북을 끝까지 실행하면, 동일 디렉터리에 `Toss_preprocessed_FF.csv`가 생성됩니다.

### 5.2. Stateless MAB 실험 노트북

1. `RL_TOSS_MAB_Stateless.ipynb`를 열어 실행합니다.  
2. 전처리 데이터 로딩 코드가 아래와 같이 되어 있는지 확인합니다.

    import pandas as pd
    df = pd.read_csv("Toss_preprocessed_FF.csv")

3. 각 알고리즘(ε-Greedy, UCB1, Thompson, Softmax)에 대해 시뮬레이션이 수행되며,  
   CTR, Regret, 팔 선택 분포 등을 그래프로 확인할 수 있습니다.

### 5.3. Contextual Bandit (LinUCB) 노트북

1. `RL_TOSS_MAB_Contextual Bandit.ipynb`를 열어 실행합니다.  
2. 동일하게 `Toss_preprocessed_FF.csv`를 로드한 뒤,  
   요일/시간/최근 노출 이력 등 컨텍스트 피처를 사용하여 LinUCB 알고리즘을 학습·평가합니다.  

---

## 6. 실험 결과 요약 

- **Baseline CTR**: 약 **0.3438**  
- **Stateless MAB 최고 성능**: Thompson Sampling  
  - 최종 CTR: 약 **0.3609**  
- **Contextual Bandit (LinUCB)**  
  -단일 seed CTR: 약 **0.3462**   
  -10 seed 평균 CTR: 약 **0.3570** (Baseline 대비 약 +1.3%p)  

→ 탐색 전략과 컨텍스트 정보를 활용함으로써,  
단순 고정 정책(Baseline)보다 광고 클릭률을 유의미하게 개선할 수 있음을 확인했습니다.


---
