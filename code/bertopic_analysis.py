import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan

# 불용어 및 NLP 도구 준비
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

custom_stopwords = {
    'sport', 'sports', 'result', 'data', 'study', 'analysis','soccer', 'football',
    'type', 'year', 'method','injury', 'injuries', 'player', 'athlete', 'team'
}

# 데이터 불러오기
file_paths = ["soccer_injury.txt", "soccer_injury (1).txt"]
all_abstracts = []
for path in file_paths:
    with open(path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
        abstracts = re.findall(r'AB (.*?)(?=(?:\n[A-Z]{2} )|\Z)', raw_text, re.DOTALL)
        all_abstracts.extend(abstracts)

# 텍스트 정제
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9.%\s]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words and word not in custom_stopwords]
    return ' '.join(words)

unique_abstracts = list(set(all_abstracts))
cleaned_abstracts = [clean_text(text) for text in unique_abstracts]
cleaned_abstracts = [text for text in cleaned_abstracts if text.strip()]

# 임베딩 및 클러스터링 구성 요소 정의
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=8, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer()
representation_model = KeyBERTInspired()

# BERTopic 모델 정의
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    nr_topics=10,
    n_gram_range=(1, 2),
    calculate_probabilities=True,
    verbose=True,
    language="english"
)

# 토픽 모델 학습 실행
topics, probs = topic_model.fit_transform(cleaned_abstracts)

# 시각화 결과 저장
try:
    fig = topic_model.visualize_topics(top_n_topics=10)
    fig.write_html("bertopic_visualization.html")
except ValueError:
    print("토픽 수가 적어 visualize_topics()를 생략합니다.")

topic_model.visualize_barchart(top_n_topics=10).write_html("bertopic_barchart.html")
topic_model.visualize_heatmap().write_html("bertopic_heatmap.html")

# 토픽 및 문서 정보 저장
topic_info = topic_model.get_topic_info()
topic_info.to_csv("bertopic_topic_info.csv", index=False)

doc_info = topic_model.get_document_info(cleaned_abstracts)
doc_info.to_csv("bertopic_doc_info.csv", index=False)

# 토픽별 핵심 키워드 저장
with open("bertopic_top_words.csv", "w", encoding="utf-8") as f:
    for topic_num in topic_model.get_topics().keys():
        words = topic_model.get_topic(topic_num)
        if words:
            line = f"Topic {topic_num}," + ",".join([w[0] for w in words]) + "\n"
            f.write(line)

print("\n시각화 및 결과 파일이 저장되었습니다.")