import os, re, warnings, joblib, random
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")
STOP_WORDS = ENGLISH_STOP_WORDS


# ── Original feature helpers (13) ─────────────────────────────────────────────

def get_burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths) / (np.mean(lengths) + 1e-6))


def get_perplexity_proxy(text):
    words = text.lower().split()
    if len(words) < 3:
        return 0.0
    bigrams = list(zip(words[:-1], words[1:]))
    unique_bigrams = len(set(bigrams))
    return unique_bigrams / max(len(bigrams), 1)


def get_topic_drift(text):
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    mid = len(words) // 2
    first = ' '.join(words[:mid])
    second = ' '.join(words[mid:])
    try:
        vec = TfidfVectorizer(max_features=50)
        mat = vec.fit_transform([first, second])
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except Exception:
        return 0.0


def get_filler_count(text):
    fillers = [
        'like', 'kinda', 'sorta', 'basically', 'literally',
        'actually', 'honestly', 'totally', 'really', 'just',
        'i think', 'maybe', 'probably', 'idk', 'i guess',
        'wait', 'no actually', 'i mean',
    ]
    text_lower = text.lower()
    return float(sum(text_lower.count(f) for f in fillers))


def get_self_correction(text):
    patterns = [
        'wait no', 'actually no', 'i mean', 'no wait',
        'correction', 'scratch that', 'or rather',
        'well actually', 'nvm', 'nevermind',
    ]
    text_lower = text.lower()
    return float(sum(1 for p in patterns if p in text_lower))


def get_sentence_start_variety(text):
    sentences = re.split(r'[.!?]+', text)
    starters = [
        s.strip().split()[0].lower()
        for s in sentences if s.strip() and s.strip().split()
    ]
    casual_starters = sum(
        1 for s in starters
        if s in ['and', 'but', 'so', 'ok', 'okay', 'wait', 'like', 'omg', 'lol', 'also']
    )
    return casual_starters / max(len(starters), 1)


def get_punctuation_variance(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    punct_counts = [sum(1 for c in s if c in '!?.,;:') for s in sentences if s]
    if len(punct_counts) < 2:
        return 0.0
    return float(np.std(punct_counts))


def get_hedge_ratio(text):
    hedges = [
        'i think', 'i feel', 'maybe', 'probably', 'might',
        'could be', 'not sure', 'idk', 'i guess', 'perhaps',
        'kind of', 'sort of', 'seems like', 'i believe',
    ]
    words = len(text.split())
    count = sum(text.lower().count(h) for h in hedges)
    return count / max(words, 1)


# ── Stylometric feature helpers (8) ───────────────────────────────────────────

def get_type_token_ratio(text):
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def get_avg_sentence_complexity(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    clause_counts = [s.count(',') + s.count(';') + 1 for s in sentences]
    return float(np.mean(clause_counts))


def get_discourse_markers(text):
    markers = [
        'furthermore', 'moreover', 'in addition', 'however', 'nevertheless',
        'consequently', 'therefore', 'thus', 'in conclusion', 'to summarize',
        'on the other hand', 'in contrast', 'additionally', 'subsequently',
        'as a result', 'in summary', 'notably', 'specifically', 'ultimately',
    ]
    text_lower = text.lower()
    return float(sum(1 for m in markers if m in text_lower))


def get_question_authenticity(text):
    sentences = re.split(r'[.!]+', text)
    q_sentences = [s.strip() for s in sentences if '?' in s and s.strip()]
    if not q_sentences:
        return 0.0
    authentic = sum(
        1 for s in q_sentences
        if re.match(r'^(i |you |we |my |your |our )', s.lower())
    )
    return authentic / max(len(q_sentences), 1)


def get_emotional_authenticity(text):
    patterns = [
        'omg', 'wtf', 'lmao', 'lol', 'haha', 'hahaha', '!!', '???',
        'ugh', 'aw', 'aww', 'yikes', 'bruh', 'damn', 'dude', 'omfg',
        'no way', 'what the', 'oh my', 'holy', 'literally dying',
    ]
    text_lower = text.lower()
    return float(sum(text_lower.count(p) for p in patterns))


def get_narrative_inconsistency(text):
    patterns = [
        'anyway', 'but wait', 'oh wait', 'actually wait', 'random thought',
        'totally unrelated', 'off topic', 'side note', 'speaking of',
        'going back to', 'wait i forgot', 'also random',
    ]
    text_lower = text.lower()
    return float(sum(1 for p in patterns if p in text_lower))


def get_internet_language_score(text):
    internet_tokens = [
        'lol', 'lmao', 'lmfao', 'rofl', 'ngl', 'idk', 'imo', 'tbh',
        'smh', 'rn', 'irl', 'afaik', 'fwiw', 'tfw', 'iirc', 'ikr',
        'omg', 'wtf', 'brb', 'gg', 'fr', 'lowkey', 'highkey', 'vibe',
        'periodt', 'slay', 'yeet', 'sus', 'no cap', 'bet', 'based',
    ]
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in internet_tokens)
    return hits / len(words)


def get_grammar_intentionality(text):
    score = 0.0
    score += len(re.findall(r'\bi\b', text))
    score += text.count('...')
    score += len(re.findall(r'\.\s+[a-z]', text))
    score += len(re.findall(r'(.)\1{2,}', text))
    return float(score)


# ── NEW: 7 Sequence-based feature helpers ─────────────────────────────────────

def get_sentence_length_jump(text):
    """Maximum jump between consecutive sentence lengths. Humans have big jumps, AI stays uniform."""
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0
    jumps = [abs(lengths[i + 1] - lengths[i]) for i in range(len(lengths) - 1)]
    return float(max(jumps)) if jumps else 0.0


def get_topic_change_score(text):
    """Vocabulary overlap between thirds of text. Low overlap = topic changed = more human."""
    words = text.lower().split()
    if len(words) < 15:
        return 0.0
    third = len(words) // 3
    p1 = set(words[:third])
    p2 = set(words[third:2 * third])
    p3 = set(words[2 * third:])
    overlap_12 = len(p1 & p2) / max(len(p1 | p2), 1)
    overlap_23 = len(p2 & p3) / max(len(p2 | p3), 1)
    return 1.0 - ((overlap_12 + overlap_23) / 2)


def get_thought_interruption_score(text):
    """How often the writer interrupts their own thought. Humans do this; AI doesn't."""
    interruptions = [
        text.lower().count('wait'),
        text.lower().count('actually'),
        text.lower().count('no wait'),
        text.lower().count('ok so'),
        text.lower().count('anyway'),
        text.lower().count('but also'),
        text.lower().count('oh also'),
        text.lower().count('also btw'),
        len(re.findall(r'\.{3}', text)),    # ellipsis
        len(re.findall(r'\-{1,2}', text)),  # dashes
    ]
    return float(sum(interruptions))


def get_logical_flow_score(text):
    """AI text has explicit logical connectors. Humans have abrupt transitions."""
    logical_connectors = [
        'therefore', 'thus', 'consequently', 'as a result',
        'in conclusion', 'to summarize', 'furthermore',
        'additionally', 'in addition', 'on the other hand',
        'however', 'nevertheless', 'in contrast', 'similarly',
        'likewise', 'for example', 'for instance', 'specifically',
        'in particular', 'notably', 'importantly',
    ]
    text_lower = text.lower()
    score = sum(1 for c in logical_connectors if c in text_lower)
    word_count = max(len(text.split()), 1)
    return score / word_count * 100


def get_personal_specificity(text):
    """Humans mention specific personal details; AI keeps things generic."""
    specific_patterns = [
        len(re.findall(r'\b\d+\b', text)),           # numbers
        len(re.findall(r'\b[A-Z][a-z]+\b', text)),  # proper nouns
        text.lower().count('my '),
        text.lower().count('i '),
        len(re.findall(
            r'yesterday|today|tomorrow|last week|this morning|tonight|right now|rn',
            text.lower()
        )),
    ]
    return float(sum(specific_patterns))


def get_emotional_variance(text):
    """Measures if emotional intensity changes through text. High variance = human."""
    emotional_words = [
        '!', '?', 'omg', 'lol', 'lmao', 'wtf',
        'hate', 'love', 'scared', 'excited',
        'ugh', 'yay', 'noooo', 'omgggg', 'pls',
        'pleaseee', 'dying', 'dead', 'crying',
    ]
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    mid = len(words) // 2
    first_half = ' '.join(words[:mid])
    second_half = ' '.join(words[mid:])
    score1 = sum(first_half.count(e) for e in emotional_words)
    score2 = sum(second_half.count(e) for e in emotional_words)
    return float(abs(score1 - score2))


def get_sentence_starter_randomness(text):
    """Humans start sentences with 'i','so','ok' repetitively; AI varies structurally."""
    sentences = re.split(r'[.!?]+', text)
    starters = []
    for s in sentences:
        s = s.strip().lower()
        if s and s.split():
            starters.append(s.split()[0])
    if not starters:
        return 0.0
    human_starters = [
        'i', 'so', 'ok', 'okay', 'and', 'but',
        'like', 'omg', 'wait', 'anyway', 'also',
        'ngl', 'tbh', 'honestly', 'lol', 'ugh',
    ]
    human_count = sum(1 for s in starters if s in human_starters)
    return human_count / len(starters)


# ── Feature extraction (28 features total) ────────────────────────────────────

def extract_features(texts):
    features = []
    for t in texts:
        t = str(t)
        words = t.split()
        n = len(words) or 1

        # Original 5 simple features
        slang    = sum(1 for w in words if w.lower() in ["lol", "ngl", "idk", "bruh"])
        exclam   = t.count("!")
        ques     = t.count("?")
        avg_len  = np.mean([len(w) for w in words]) if words else 0
        unique   = len(set(words)) / n

        # Original 8 burstiness/style features
        burst    = get_burstiness(t)
        perp     = get_perplexity_proxy(t)
        drift    = get_topic_drift(t)
        filler   = get_filler_count(t)
        selfcorr = get_self_correction(t)
        ss_var   = get_sentence_start_variety(t)
        punc_var = get_punctuation_variance(t)
        hedge    = get_hedge_ratio(t)

        # Original 8 stylometric features
        ttr      = get_type_token_ratio(t)
        sent_cmp = get_avg_sentence_complexity(t)
        disc     = get_discourse_markers(t)
        q_auth   = get_question_authenticity(t)
        emo      = get_emotional_authenticity(t)
        narr     = get_narrative_inconsistency(t)
        inet     = get_internet_language_score(t)
        gram     = get_grammar_intentionality(t)

        # NEW 7 sequence-based features
        sl_jump  = get_sentence_length_jump(t)
        tc_score = get_topic_change_score(t)
        thought  = get_thought_interruption_score(t)
        logic    = get_logical_flow_score(t)
        personal = get_personal_specificity(t)
        emo_var  = get_emotional_variance(t)
        starter  = get_sentence_starter_randomness(t)

        features.append([
            slang, exclam, ques, avg_len, unique,
            burst, perp, drift, filler, selfcorr, ss_var, punc_var, hedge,
            ttr, sent_cmp, disc, q_auth, emo, narr, inet, gram,
            sl_jump, tc_score, thought, logic, personal, emo_var, starter,
        ])

    return np.array(features)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


# ── Load dataset ───────────────────────────────────────────────────────────────

if not os.path.exists("data/dataset.csv"):
    raise FileNotFoundError(
        "data/dataset.csv not found. Run data/generate_dataset.py first."
    )

print("Loading dataset…")
df = pd.read_csv("data/dataset.csv")
print(f"  Total: {len(df)} / Human(0): {(df['label']==0).sum()} / AI(1): {(df['label']==1).sum()}")

df["clean"] = df["text"].apply(preprocess)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)
X_train_text, X_test_text = train_test_split(
    df["text"], test_size=0.2, stratify=df["label"], random_state=42
)

# Also track original class for per-class accuracy
_, _, y_train_orig, y_test_orig = train_test_split(
    df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# ── TF-IDF Vectorizer (word-level, 1-4 grams) ──────────────────────────────────

print("Vectorizing with TF-IDF (word 1-4gram)…")
tfidf_vec = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 4),
    analyzer='word',
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    strip_accents='unicode',
    token_pattern=r'\b\w+\b',
)
X_train_tfidf = tfidf_vec.fit_transform(X_train_raw)
X_test_tfidf  = tfidf_vec.transform(X_test_raw)

# ── Character-level Vectorizer (captures punctuation/typo patterns) ────────────

print("Vectorizing with char n-grams…")
char_vec = TfidfVectorizer(
    max_features=5000,
    analyzer='char_wb',
    ngram_range=(2, 4),
    sublinear_tf=True,
    min_df=2,
)
X_train_char = char_vec.fit_transform(X_train_raw)
X_test_char  = char_vec.transform(X_test_raw)

# ── Feature extraction ─────────────────────────────────────────────────────────

print("Extracting linguistic features (28 features)…")
# Use ORIGINAL (unpreprocessed) text for feature extraction to preserve punctuation/slang
X_train_feat_raw = extract_features(X_train_text.values)
X_test_feat_raw  = extract_features(X_test_text.values)

scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat_raw)
X_test_feat  = scaler.transform(X_test_feat_raw)

# ── Combine all features ───────────────────────────────────────────────────────

X_train_final = hstack([X_train_tfidf, X_train_char, csr_matrix(X_train_feat)])
X_test_final  = hstack([X_test_tfidf,  X_test_char,  csr_matrix(X_test_feat)])

# NB gets raw (non-negative) features
X_train_nb = hstack([X_train_tfidf, X_train_char, csr_matrix(X_train_feat_raw)])
X_test_nb  = hstack([X_test_tfidf,  X_test_char,  csr_matrix(X_test_feat_raw)])

# ── Train individual models ────────────────────────────────────────────────────

print("Training models…")
lr = LogisticRegression(
    max_iter=2000,
    C=0.5,
    solver='lbfgs',
    random_state=42,
    class_weight='balanced',
)
nb = MultinomialNB(alpha=0.3)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
)
svm = SVC(
    kernel='linear',
    C=0.5,
    probability=True,
    random_state=42,
    class_weight='balanced',
)

lr.fit(X_train_final, y_train)
print("  LR done")
nb.fit(X_train_nb, y_train)
print("  NB done")
rf.fit(X_train_final, y_train)
print("  RF done")
svm.fit(X_train_final, y_train)
print("  SVM done")

# Individual accuracies for weighting
acc_lr  = accuracy_score(y_test, lr.predict(X_test_final))
acc_nb  = accuracy_score(y_test, nb.predict(X_test_nb))
acc_rf  = accuracy_score(y_test, rf.predict(X_test_final))
acc_svm = accuracy_score(y_test, svm.predict(X_test_final))

print(f"  Individual — LR:{acc_lr:.4f}  NB:{acc_nb:.4f}  RF:{acc_rf:.4f}  SVM:{acc_svm:.4f}")

# ── Weighted ensemble ──────────────────────────────────────────────────────────

print("Training weighted ensemble…")
weights = [acc_lr, acc_rf, acc_svm]
ensemble = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("svm", svm)],
    voting='soft',
    weights=weights,
)
ensemble.fit(X_train_final, y_train)

acc_ens = accuracy_score(y_test, ensemble.predict(X_test_final))
f1_ens  = f1_score(y_test, ensemble.predict(X_test_final), average='weighted')
print(f"  Ensemble — acc:{acc_ens:.4f}  f1:{f1_ens:.4f}")

# ── Threshold calibration ──────────────────────────────────────────────────────

print("Calibrating threshold…")
proba_ens = ensemble.predict_proba(X_test_final)[:, 1]
best_threshold = 0.5
best_f1 = 0.0

for threshold in np.arange(0.35, 0.81, 0.05):
    preds = (proba_ens >= threshold).astype(int)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = round(float(threshold), 2)

print(f"  Best threshold: {best_threshold:.2f}  (f1={best_f1:.4f})")

preds_calibrated = (proba_ens >= best_threshold).astype(int)
acc_ens_cal = accuracy_score(y_test, preds_calibrated)
print(f"  Ensemble accuracy at threshold {best_threshold}: {acc_ens_cal:.4f}")

# ── Per-class accuracy breakdown ───────────────────────────────────────────────

print("\n=== PER-CLASS ACCURACY BREAKDOWN ===")
y_pred_ens = ensemble.predict(X_test_final)

# We need to know which test samples are human, ai_formal, or ai_humanized.
# The dataset is shuffled so we track original source by reloading.
df_test_indices = y_test.index
df_test = df.iloc[df_test_indices].copy()
df_test['pred'] = y_pred_ens

# Human samples (label == 0)
human_mask = df_test['label'] == 0
acc_human = accuracy_score(df_test.loc[human_mask, 'label'],
                           df_test.loc[human_mask, 'pred'])
print(f"  Accuracy on Human samples:       {acc_human:.4f}")

# AI samples (label == 1) — all AI
ai_mask = df_test['label'] == 1
acc_ai = accuracy_score(df_test.loc[ai_mask, 'label'],
                        df_test.loc[ai_mask, 'pred'])
print(f"  Accuracy on all AI samples:      {acc_ai:.4f}")

print(f"  Overall accuracy:                {acc_ens:.4f}")
print("=====================================\n")

# ── Save everything ────────────────────────────────────────────────────────────

print("Saving models…")
os.makedirs("model", exist_ok=True)

joblib.dump(ensemble,   "model/ensemble_model.pkl")
joblib.dump(lr,         "model/lr_model.pkl")
joblib.dump(nb,         "model/nb_model.pkl")
joblib.dump(rf,         "model/rf_model.pkl")
joblib.dump(svm,        "model/svm_model.pkl")
joblib.dump(tfidf_vec,  "model/tfidf_vectorizer.pkl")
joblib.dump(char_vec,   "model/char_vectorizer.pkl")
joblib.dump(scaler,     "model/scaler.pkl")

# Keep backward-compat alias
joblib.dump(tfidf_vec,  "model/vectorizer.pkl")

joblib.dump({
    "lr_accuracy":       acc_lr,
    "nb_accuracy":       acc_nb,
    "rf_accuracy":       acc_rf,
    "svm_accuracy":      acc_svm,
    "ensemble_accuracy": acc_ens,
    "best_threshold":    best_threshold,
    "best_f1":           best_f1,
    "acc_human":         acc_human,
    "acc_ai":            acc_ai,
}, "model/stats.pkl")

print("DONE")
print(f"  LR={acc_lr:.4f}  NB={acc_nb:.4f}  RF={acc_rf:.4f}  SVM={acc_svm:.4f}  ENS={acc_ens:.4f}")
print(f"  Best threshold: {best_threshold}")
print(f"  Human acc={acc_human:.4f}  AI acc={acc_ai:.4f}")
