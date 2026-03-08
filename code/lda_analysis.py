import numpy as np
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# 불용어 및 NLP 도구 준비
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

custom_stopwords = {
    'player', 'athlete', 'team', 'sport', 'result', 'data', 'study', 'analysis', 'p',
    'performance', 'group', 'method', 'based', 'high', 'low', 'level', 'rate', 'year', 'use', 'type'
}

# 데이터 불러오기
file_paths = ["soccer_injury.txt", "soccer_injury (1).txt"]
all_abstracts = []
for path in file_paths:
    with open(path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
        abstracts = re.findall(r'AB (.*?)(?=(?:\n[A-Z]{2} )|\Z)', raw_text, re.DOTALL)
        all_abstracts.extend(abstracts)

unique_abstracts = list(set(all_abstracts))

# 텍스트 정제
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words and w not in custom_stopwords])

cleaned_abstracts = [clean_text(text) for text in unique_abstracts]

# (Tokenization + Lemmatization)
tokenizer = RegexpTokenizer(r'\d+\.\d+%|\d+%|\d+\.\d+|\d+|[A-Za-z]+\.[A-Za-z]+|\w+')
tokenized_abstracts = [tokenizer.tokenize(text) for text in cleaned_abstracts]

def lemmatize_tokens(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

lemmatized_abstracts = [lemmatize_tokens(tokens) for tokens in tokenized_abstracts]

# 전처리 결과를 데이터프레임으로 정리 (저장은 필요 시 선택적으로 수행)
df = pd.DataFrame({
    "original_abstract": unique_abstracts,
    "cleaned_text": cleaned_abstracts,
    "tokens": tokenized_abstracts,
    "lemmatized_tokens": lemmatized_abstracts
})

# LDA 실행
texts = df['lemmatized_tokens'].tolist()
dictionary = corpora.Dictionary(texts)
corpus_bow = [dictionary.doc2bow(text) for text in texts]
tfidf_model = models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf_model[corpus_bow]

lda_model = models.LdaModel(
    corpus=corpus_bow,
    id2word=dictionary,
    num_topics=4,
    random_state=42,
    update_every=1,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

# 토픽 출력
print("주요 토픽:")
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"Topic {idx}: {topic}")

# 시각화 파일 저장
vis = gensimvis.prepare(lda_model, corpus_tfidf, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')
print("'lda_visualization.html' 파일이 생성되었습니다.")