from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import numpy as np
import tensorflow as tf

# Initialize Flask
app = Flask(__name__)

# Load trained model
model = joblib.load("quora_best_model.pkl")

# Load SBERT model (TensorFlow backend)
tf.get_logger().setLevel('ERROR')  # suppress TF warnings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocessing
def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent ').replace('$', ' dollar ').replace('₹', ' rupee ')
    q = q.replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '')
    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q).strip()
    return q

# Longest common substring ratio
def longest_substr_ratio(s1, s2):
    m = [[0]*(1+len(s2)) for _ in range(1+len(s1))]
    longest = 0
    for i in range(1,len(s1)+1):
        for j in range(1,len(s2)+1):
            if s1[i-1] == s2[j-1]:
                m[i][j] = m[i-1][j-1] + 1
                if m[i][j] > longest:
                    longest = m[i][j]
            else:
                m[i][j] = 0
    return longest / max(len(s1), len(s2), 1)

# Feature extraction
def build_features(q1, q2):
    words1, words2 = q1.split(), q2.split()
    len1, len2 = len(q1), len(q2)
    
    features = {}
    
    # Length-based features
    features['abs_len_diff'] = abs(len1 - len2)
    features['mean_len'] = (len1 + len2)/2
    
    # First/last word match
    features['first_word_eq'] = int(words1[0] == words2[0]) if words1 and words2 else 0
    features['last_word_eq'] = int(words1[-1] == words2[-1]) if words1 and words2 else 0
    
    # Fuzzy features
    features['fuzz_ratio'] = fuzz.ratio(q1,q2)
    features['fuzz_partial_ratio'] = fuzz.partial_ratio(q1,q2)
    features['token_sort_ratio'] = fuzz.token_sort_ratio(q1,q2)
    features['token_set_ratio'] = fuzz.token_set_ratio(q1,q2)
    
    # Character/word set stats
    def char_word_stats(s1,s2):
        c1 = list(s1.replace(' ',''))
        c2 = list(s2.replace(' ',''))
        csc = [len(set(c1)&set(c2))/max(1,len(set(c1)|set(c2)))]  # char set ratio
        cwc = [len(set(words1)&set(words2))/max(1,len(set(words1)|set(words2)))]  # word set ratio
        ctc = [len(c1)+len(c2)]  # total chars
        return csc, cwc, ctc
    
    csc, cwc, ctc = char_word_stats(q1,q2)
    features['csc_min'] = np.min(csc)
    features['csc_max'] = np.max(csc)
    features['cwc_min'] = np.min(cwc)
    features['cwc_max'] = np.max(cwc)
    features['ctc_min'] = np.min(ctc)
    features['ctc_max'] = np.max(ctc)
    
    # Longest substring ratio
    features['longest_substr_ratio'] = longest_substr_ratio(q1, q2)
    
    # SBERT embeddings
    emb1 = sbert_model.encode(q1, convert_to_numpy=True)
    emb2 = sbert_model.encode(q2, convert_to_numpy=True)
    features['sbert_cosine'] = float(np.dot(emb1,emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2)))
    features['sbert_dot'] = float(np.dot(emb1,emb2))
    features['sbert_absdiff'] = float(np.sum(np.abs(emb1-emb2)))
    
    return features

# Feature order to match model
FEATURE_ORDER = ['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
                 'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
                 'longest_substr_ratio', 'token_sort_ratio', 'token_set_ratio',
                 'fuzz_ratio', 'fuzz_partial_ratio', 'sbert_cosine', 'sbert_absdiff', 'sbert_dot']

# Web interface
@app.route("/", methods=["GET","POST"])
def home():
    result = None
    if request.method=="POST":
        q1 = preprocess(request.form.get("q1",""))
        q2 = preprocess(request.form.get("q2",""))
        features = build_features(q1,q2)
        X_input = pd.DataFrame([features])[FEATURE_ORDER]
        pred = model.predict(X_input)[0]
        result = "Duplicate ✅" if pred else "Not Duplicate ❌"
    
    html = """
    <!doctype html>
    <html>
    <head><title>Quora Question Pair Checker</title></head>
    <body>
    <h2>Quora Question Pair Duplicate Checker</h2>
    <form method="post">
      Question 1:<br><input type="text" name="q1" size="60"><br><br>
      Question 2:<br><input type="text" name="q2" size="60"><br><br>
      <input type="submit" value="Check Duplicate">
    </form>
    {% if result %}<h3>Result: {{ result }}</h3>{% endif %}
    </body>
    </html>
    """
    return render_template_string(html, result=result)

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        q1 = preprocess(data.get("q1",""))
        q2 = preprocess(data.get("q2",""))
        features = build_features(q1,q2)
        X_input = pd.DataFrame([features])[FEATURE_ORDER]
        pred = model.predict(X_input)[0]
        return jsonify({"duplicate": bool(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__=="__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
