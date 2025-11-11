# app.py (updated)
import joblib, os, numpy as np, pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/svm_tfidf_pipeline.joblib" 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH + " not found. Train model first.")

model = joblib.load(MODEL_PATH)  

 
def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)

# chuẩn hóa nhãn dự đoán nếu cần thiết (VN->EN hoặc EN->EN)
vn_to_en = {
    'Vui': 'Enjoyment','Vui vẻ':'Enjoyment','Buồn':'Sadness','Giận':'Anger',
    'Ghê tởm':'Disgust','Sợ hãi':'Fear','Ngạc nhiên':'Surprise','Khác':'Other'
}
en_set = set(['Enjoyment','Sadness','Anger','Disgust','Fear','Surprise','Other'])

def normalize_label_out(lbl):
    lbl = str(lbl).strip()
    if lbl in en_set:
        return lbl
    return vn_to_en.get(lbl, lbl)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text","").strip()
    if text=="":
        return jsonify({"error":"No text provided"}), 400

    if hasattr(model, "predict"):
        pred = model.predict([text])[0]
    else:
        return jsonify({"error":"Model invalid"}), 500

    
    clf = None
    if hasattr(model, "named_steps"):
        # pipeline
        steps = model.named_steps
        clf = steps[list(steps.keys())[-1]]
    else:
        clf = model

    probs = None
    labels = None
    try:
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(model.named_steps['tfidf'].transform([text])) if hasattr(model, "named_steps") else clf.decision_function([text])
            scores = np.atleast_2d(scores)
            probs = softmax(scores).tolist()[0]
            if hasattr(clf, "classes_"):
                labels = clf.classes_.tolist()
        elif hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(model.named_steps['tfidf'].transform([text])).tolist()[0]
            labels = clf.classes_.tolist()
    except Exception:
        probs = None

    pred_norm = normalize_label_out(pred)
    confidence = None
    if probs and labels:
        
        try:
            idx = labels.index(pred)
            confidence = float(probs[idx]*100)
        except:
            confidence = max([p*100 for p in probs])
    else:
        confidence = 85.0

    return jsonify({
        "prediction": pred_norm,
        "raw_prediction": str(pred),
        "labels": labels or [],
        "probs": probs or [],
        "confidence": round(confidence,2)
    })

# analyze_all: đọc 3 excel, predict trên toàn bộ data, trả metrics + small sample
@app.route("/analyze_all", methods=["GET"])
def analyze_all():
    files = ['train_nor_811.xlsx','valid_nor_811.xlsx','test_nor_811.xlsx']
    dfs = [pd.read_excel(f) for f in files if os.path.exists(f)]
    if not dfs:
        return jsonify({"error":"No dataset files found"}), 400
    df = pd.concat(dfs, ignore_index=True)
    # detect cols
    cols = df.columns.tolist()
    text_col = next((c for c in cols if 'sent' in c.lower() or 'text' in c.lower() or 'comment' in c.lower()), cols[0])
    label_col = next((c for c in cols if 'emotion' in c.lower() or 'label' in c.lower()), cols[-1])
    df = df[[text_col, label_col]].dropna()
    texts = df[text_col].astype(str).tolist()
    true = df[label_col].astype(str).tolist()

    preds = model.predict(texts).tolist()
    preds_norm = [normalize_label_out(p) for p in preds]
    true_norm = [normalize_label_out(t) for t in true]

    acc = accuracy_score(true_norm, preds_norm)
    f1w = f1_score(true_norm, preds_norm, average='weighted')
    report = classification_report(true_norm, preds_norm, output_dict=True)
    cm = confusion_matrix(true_norm, preds_norm, labels=list(en_set)).tolist()

    sample = []
    for i in range(min(100, len(texts))):
        sample.append({"text": texts[i], "gold": true[i], "pred": preds[i]})

    return jsonify({
        "total": len(texts),
        "accuracy": round(acc*100,3),
        "f1_weighted": round(f1w*100,3),
        "report": report,
        "confusion_matrix": cm,
        "labels": list(en_set),
        "sample": sample[:50]
    })

if __name__ == "__main__":
    print("Running Flask API on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
