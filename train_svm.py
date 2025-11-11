# train_svm.py (snippet cập nhật)
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
import joblib
import os

# mapping VN -> EN (nếu file của bạn có nhãn VN)
vn_to_en = {
    'Vui': 'Enjoyment', 'Vui vẻ': 'Enjoyment', 'Enjoyment':'Enjoyment',
    'Buồn': 'Sadness', 'Sadness':'Sadness',
    'Giận': 'Anger', 'Anger':'Anger',
    'Ghê tởm': 'Disgust', 'Ghê': 'Disgust', 'Disgust':'Disgust',
    'Sợ hãi': 'Fear', 'Sợ': 'Fear', 'Fear':'Fear',
    'Ngạc nhiên': 'Surprise', 'Surprise':'Surprise',
    'Khác': 'Other', 'Other':'Other'
}

def normalize_label(x):
    x = str(x).strip()
    return vn_to_en.get(x, x)

# load & concat files (your existing logic)
df = pd.concat([pd.read_excel(f) for f in ['train_nor_811.xlsx','valid_nor_811.xlsx','test_nor_811.xlsx']], ignore_index=True)

# detect text/label columns as before; suppose they are tcol, lcol
tcol = [c for c in df.columns if 'sent' in c.lower() or 'text' in c.lower() or 'comment' in c.lower()][0]
lcol = [c for c in df.columns if 'emotion' in c.lower() or 'label' in c.lower()][0]

df = df[[tcol,lcol]].dropna()
df[tcol] = df[tcol].astype(str)
df[lcol] = df[lcol].astype(str).map(normalize_label)   # **chuẩn hoá nhãn ở đây**

X = df[tcol].values
y = df[lcol].values

# build pipeline and gridsearch as before, but save the full pipeline:
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=30000)),
    ('clf', LinearSVC(max_iter=5000))
])
params = {'clf__C': [0.1, 1.0, 5.0]}
gs = GridSearchCV(pipeline, params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
gs.fit(X, y)

best = gs.best_estimator_
os.makedirs('models', exist_ok=True)
joblib.dump(best, 'models/svm_tfidf_pipeline.joblib')   # LƯU *pipeline* (TF-IDF + SVM) để dễ dùng
print("Saved models/svm_tfidf_pipeline.joblib")
