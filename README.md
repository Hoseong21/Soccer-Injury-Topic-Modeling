# 토픽 모델링을 통한 축구 부상 유형 및 위험 요인 탐색
### Soccer Injury Topic Modeling with LDA & BERTopic

축구 선수 부상 관련 학술 문헌의 초록(Abstract) 데이터를 활용하여, LDA와 BERTopic 기법으로 주요 부상 유형 및 위험 요인을 도출한 텍스트 마이닝 개인 과제입니다.

<br>

## 📌 Project Info

| **기간** | 2025.04.02 – 2025.05.19 |
|---|---|
| **분류** | 개인 프로젝트 |
| **수업** | 텍스트마이닝 |
| **데이터** | Web of Science — 축구 부상 관련 학술 문헌 초록 |

<br>

## 📖 Overview

본 프로젝트는 **Web of Science**에서 수집한 축구 선수 부상 관련 학술 문헌의 초록(Abstract) 데이터를 활용하여, 반복적으로 등장하는 **부상 유형과 위험 요인**을 토픽 모델링으로 탐색한 분석입니다.

최근 프로 축구 대회의 경기 수 증가에 따라 선수 부상이 빈번해지는 추세에 주목하였으며, 통계 기반 모델인 **LDA**와 의미 기반 임베딩 모델인 **BERTopic**을 병행하여 전통적인 부상 유형부터 심리적 요인, 장비 환경 등 비정형적 위험 요인까지 입체적으로 분석하였습니다.

<br>

## ✨ Key Contributions

- Web of Science에서 수집한 **1,194편의 문헌** 중 중복 제거 후 **1,124개의 유효한 초록**을 최종 분석 데이터로 활용하였습니다.
- LDA와 BERTopic의 **상호보완적 특성**을 고려하여 각 기법에 최적화된 전처리 파이프라인을 별도로 설계하였습니다.
- LDA는 **BoW + TF-IDF** 기반으로 전체적인 토픽 흐름을 파악하고, BERTopic은 **SentenceTransformer 임베딩 + UMAP + HDBSCAN**으로 의미 기반 클러스터링을 수행하였습니다.
- BERTopic을 통해 LDA로는 분리되기 어려운 **뇌진탕, 심리적 스트레스, 신발 착용 환경** 등의 비정형적 위험 요인까지 포착하였습니다.
- 두 모델의 결과를 **ChatGPT 기반 토픽 도출 결과와 비교**하여, LDA는 부상 유형, BERTopic은 부상 위험 요인과 높은 유사성을 보임을 확인하였습니다.
- 두 모델의 결과를 **ChatGPT 기반 토픽 도출 결과와 비교**하여, LDA는 부상 유형, BERTopic은 부상 위험 요인과 높은 유사성을 보임을 확인하였습니다.

<br>

## 🗂 Dataset

**출처** : Web of Science

**검색 조건**

> TS = (soccer OR "association football") AND TS = ("injury risk" OR "injury type") NOT TS = ("American football" OR rugby)

| 단계 | 문헌 수 |
|---|---|
| 초기 수집 | 1,211편 |
| 초록 포함 문헌 선별 | 1,194편 |
| 중복 제거 후 최종 분석 데이터 | 1,124개 |

<br>

## ⚙️ Preprocessing

LDA와 BERTopic의 특성 차이를 고려하여 각 기법에 맞는 전처리 파이프라인을 별도로 구성하였습니다.

### 1. LDA 전처리

| 과정 | 내용 |
|---|---|
| 소문자 변환 | 동일한 의미의 단어를 정제 |
| 특수문자 정제 | 정규표현식으로 불필요한 기호 제거 |
| 불용어 제거 | NLTK 불용어 리스트 + 커스텀 불용어 |
| 토큰화 | `RegexpTokenizer` 활용 |
| 표제어 추출 | spaCy `en_core_web_sm` 모델 활용 |
| BoW / TF-IDF | 단어 빈도 기반 가중치 조정 |

### 2. BERTopic 전처리

| 과정 | 내용 |
|---|---|
| 소문자 변환 | 동일한 의미의 단어를 정제 |
| 특수문자 정제 | 수치 기반 표현(소수점, %) 유지 |
| 불용어 제거 | NLTK 불용어 리스트 + 커스텀 불용어 |
| 문장 임베딩 | `all-MiniLM-L6-v2` (SentenceTransformer) — 문장 단위 의미 벡터 변환 (토큰화·표제어 추출 생략) |
| 차원 축소 | UMAP (`n_neighbors=15, n_components=5`) |
| 클러스터링 | HDBSCAN (`min_cluster_size=8`) |
| 키워드 추출 | CountVectorizer + ClassTF-IDF + KeyBERTInspired |

<br>

## 📊 Results

### LDA 분석 결과 (4 Topics)

| 토픽 | 주요 키워드 | 해석 |
|---|---|---|
| Topic 1 | injury, incidence, review, turf, grass, risk | 부상 발생률 및 운동장 환경 요인 |
| Topic 2 | knee, jump, abduction, landing, hip, angle, flexion, biomechanic | 무릎 관절 부상 메커니즘 및 생체역학 분석 |
| Topic 3 | load, workload, training, rpe, sleep, intensity, sprint | 훈련 부하 및 컨디션 관리와 부상 위험 |
| Topic 4 | hamstring, muscle, quadriceps, torque, eccentric, isokinetic, emg, imbalance | 근골격계 부상 유형 및 근력 불균형 평가 |

### BERTopic 분석 결과 (9 Topics)

| 토픽 | 주요 키워드 | 해석 |
|---|---|---|
| Topic 0 | ligament, knee, hamstrings | 무릎 및 인대 부상 |
| Topic 1 | injuries, ankle, athletes | 일반적인 운동 손상 및 신체 부위 |
| Topic 2 | training, exertion, workload | 훈련 강도 및 부하 관련 위험 요인 |
| Topic 3 | concussion, headgear, head | 뇌진탕 및 두부 손상 |
| Topic 4 | stressors, anxiety, psychological | 심리적 스트레스 및 정신적 요인 |
| Topic 5 | turf, grass, footballers | 운동장 표면 관련 요인 |
| Topic 6 | fitness, svm, predicting | 예측 모델링과 선수 상태 평가 |
| Topic 7 | training, maturity, assessment | 신체 성장 및 트레이닝 평가 |
| Topic 8 | footwear, traction, insoles | 신발 착용 및 발바닥 마찰 요인 |

### 결과 요약

- **LDA** : 명확한 주제 경계로 전체 문헌의 흐름 파악에 강점을 보였으며, **부상 유형** 분류와 높은 유사성을 보임
- **BERTopic** : 의미 기반 임베딩으로 정성적 요인까지 포착하였으며, **부상 위험 요인** 분류와 높은 유사성을 보임
- 두 모델을 병행함으로써 핵심 부상 주제를 공통적으로 확인하면서도, 각 모델이 서로 다른 각도의 인사이트를 보완적으로 제공함을 확인

<br>

## 🗃 Project Structure

```
soccer-injury-topic-modeling/
│
├── lda_analysis.py          # LDA 분석 및 pyLDAvis 시각화 생성
├── bertopic_analysis.py     # BERTopic 분석, 시각화, CSV 결과 저장
│
├── requirements.txt
└── README.md
```

> 원본 데이터(`soccer_injury.txt`)는 Web of Science 저작권 정책에 따라 레포지토리에 포함하지 않았습니다.
> Dataset 섹션의 검색 조건을 참고하여 동일한 데이터를 직접 수집할 수 있습니다.

<br>

## 🛠 Tech Stack

**Core** : Python · pandas · numpy

**Topic Modeling** : Gensim (LDA) · BERTopic

**NLP / Preprocessing** : NLTK · spaCy · scikit-learn

**Embedding** : SentenceTransformers (`all-MiniLM-L6-v2`)

**Clustering** : UMAP · HDBSCAN

**Visualization** : pyLDAvis · Plotly (BERTopic 내장)