# Toss 광고 클릭 최적화 프로젝트 (강화학습 수업)
Multi-Armed Bandit 및 Contextual Bandit 기반 토스 앱 광고 추천 정책 설계와 성능 평가
강화학습(Multi-Armed Bandit, Contextual Bandit)을 이용하여 Toss 앱 광고의 CTR(Click-Through Rate)을 최적화한 프로젝트입니다.  
이 저장소는 수업 프로젝트 제출을 위해 작성되었으며, 코드와 학습된 모델, 보고서(PPT&PDF)를 포함합니다.

---

## 1. 팀 정보

- 과목: 강화학습의 기초
- 팀장 & 팀원: A71030 배정언

---

## 2. 프로젝트 개요

- **문제 정의**  
  - Toss 앱 광고 슬롯에 어떤 광고(arm)를 노출할 때 CTR을 최대화할 것인가?
- **접근 방법**
  - Stateless MAB 알고리즘:  
    - ε-Greedy (fixed/decay)  
    - UCB1  
    - Thompson Sampling  
    - Softmax (고정형 / 감쇠형)
  - Contextual Bandit:  
    - LinUCB (요일, 시간대, 최근 노출 이력 등 컨텍스트 사용)

---

## 3. 폴더 구조

```text
.
├── data/
│   └── Toss_preprocessed_FF.csv
├── notebooks/
│   ├── Toss_Data preprocessing_EDA.ipynb
│   ├── RL_TOSS_MAB_Stateless.ipynb
│   └── RL_TOSS_MAB_Contextual Bandit.ipynb
├── report/
│   └── toss_rl_project.pptx
└── README.md
