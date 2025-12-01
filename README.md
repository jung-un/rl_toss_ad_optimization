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

```text
.
├── Toss_Data preprocessing_EDA.ipynb
├── RL_TOSS_MAB_Stateless.ipynb
├── RL_TOSS_MAB_Contextual Bandit.ipynb
├── Toss_preprocessed_FF.csv
├── sampled_500k_stratified.csv (Google Drive 링크로 제공)
└── README.md

Toss_Data preprocessing_EDA.ipynb

원본 데이터(sampled_500k_stratified.csv)를 불러와
EDA, 피처 엔지니어링, 필터링 등을 수행하고
최종 학습용 데이터인 Toss_preprocessed_FF.csv를 생성하는 노트북입니다.

RL_TOSS_MAB_Stateless.ipynb

전처리된 데이터(Toss_preprocessed_FF.csv)를 기반으로
ε-Greedy, UCB1, Thompson Sampling, Softmax 등 Stateless Multi-Armed Bandit 알고리즘을 구현하고
각 알고리즘의 CTR, Regret, 팔 선택 분포를 비교합니다.

RL_TOSS_MAB_Contextual Bandit.ipynb

동일한 전처리 데이터를 사용하여
요일/시간/최근 노출 이력 등 컨텍스트를 반영한 Contextual Bandit (LinUCB) 알고리즘을 구현하고
Baseline 및 Stateless MAB와 성능을 비교합니다.

Toss_preprocessed_FF.csv

실제 밴딧 실험에 사용되는 최종 전처리 데이터셋입니다.

용량이 약 20MB 수준으로, GitHub 저장소에 포함되어 있습니다.

## 3. 데이터
3.1. 원본 데이터 (대용량, Google Drive 제공)

파일명: sampled_500k_stratified.csv

설명: Toss 광고 로그에서 **층화 샘플링(stratified sampling)**으로 추출한 약 50만 행 규모의 데이터셋입니다.

용량이 커서 GitHub에 직접 업로드하기 어려워, Google Drive 링크로 제공합니다.

🔗 원본 데이터 다운로드 링크
👉 sampled_500k_stratified.csv (Google Drive)

(위 링크를 실제 본인 드라이브 공유 링크로 수정해 주세요.)

Colab에서 이 파일을 사용하려면,

위 링크에서 파일을 다운로드하여 Colab 작업 디렉터리에 업로드하거나

Google Drive를 마운트한 후 해당 경로를 지정해서 사용합니다.

예시:

import pandas as pd

# (예시) Colab 노트북과 같은 위치에 csv를 둔 경우
df_raw = pd.read_csv("sampled_500k_stratified.csv")

3.2. 전처리 데이터 (Toss_preprocessed_FF.csv)

Toss_Data preprocessing_EDA.ipynb를 통해 생성된 최종 학습용 데이터셋입니다.

GitHub 저장소에 포함되어 있으며, 밴딧 실험 노트북에서 바로 사용할 수 있습니다.

예시:

import pandas as pd

df = pd.read_csv("Toss_preprocessed_FF.csv")
