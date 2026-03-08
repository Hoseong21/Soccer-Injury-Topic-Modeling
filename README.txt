# 토픽 모델링 프로젝트 - 축구 부상 분석

이 프로젝트는 축구 부상 관련 학술 문헌의 초록(Abstract) 데이터를 활용하여,
LDA와 BERTopic 기법을 이용해 주요 부상 유형 및 요인을 도출하는 텍스트 마이닝 분석입니다.

## 실행 방법

1. 가상환경 생성 및 실행
```bash
python -m venv nlp_env
source nlp_env/bin/activate  # Windows의 경우: nlp_env\Scripts\activate
```

2. 필요한 라이브러리 설치
```bash
pip install -r requirements.txt
```

3. spaCy 영어 모델 설치
```bash
python -m spacy download en_core_web_sm
```

4. 분석 코드 실행
```bash
python lda_analysis.py
python bertopic_analysis.py
```

## 주요 파일 설명

- `lda_analysis.py` : LDA 기반 분석 및 시각화(html) 생성
- `bertopic_analysis.py` : BERTopic 기반 분석, 시각화(html), 결과 CSV 생성
- `*.html` : 토픽 시각화 결과
- `*.csv` : 토픽별 키워드 및 문서-토픽 매핑 정보
- `soccer_injury.txt`, `soccer_injury (1).txt` : 분석에 사용된 원본 데이터

## 참고 사항

- 코드 실행 전, 위의 spaCy 모델 다운로드 명령어를 반드시 한 번 실행해야 합니다.
- 데이터 파일은 코드와 동일한 폴더에 있어야 합니다.