from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

print("Loading models…")

# LOAD MODELS
ensemble    = joblib.load("model/ensemble_model.pkl")
lr          = joblib.load("model/lr_model.pkl")
nb          = joblib.load("model/nb_model.pkl")
rf          = joblib.load("model/rf_model.pkl")
svm         = joblib.load("model/svm_model.pkl")
tfidf_vec   = joblib.load("model/tfidf_vectorizer.pkl")
char_vec    = joblib.load("model/char_vectorizer.pkl")
scaler      = joblib.load("model/scaler.pkl")

# Backward compat: also try old name
try:
    _old_vec = joblib.load("model/vectorizer.pkl")
except Exception:
    _old_vec = tfidf_vec

# Load calibrated threshold (falls back to 0.5 if not present)
try:
    _stats = joblib.load("model/stats.pkl")
    BEST_THRESHOLD = float(_stats.get("best_threshold", 0.5))
except Exception:
    BEST_THRESHOLD = 0.5

print(f"All models loaded  (threshold={BEST_THRESHOLD})")


# ─────────────────────────────────────────────
# ORIGINAL LINGUISTIC FEATURE HELPERS (13)
# ─────────────────────────────────────────────

def get_burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.5  # neutral baseline for short text
    return float(np.std(lengths) / (np.mean(lengths) + 1e-6))


def get_perplexity_proxy(text):
    words = text.lower().split()
    if len(words) < 3:
        return 0.5  # neutral baseline for short text
    bigrams = list(zip(words[:-1], words[1:]))
    unique_bigrams = len(set(bigrams))
    return unique_bigrams / max(len(bigrams), 1)


def get_topic_drift(text):
    words = text.lower().split()
    if len(words) < 10:
        return 0.5  # neutral baseline for short text
    mid = len(words) // 2
    first = ' '.join(words[:mid])
    second = ' '.join(words[mid:])
    try:
        vec = TfidfVectorizer(max_features=50)
        mat = vec.fit_transform([first, second])
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except Exception:
        return 0.5


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


# ─────────────────────────────────────────────
# STYLOMETRIC FEATURE HELPERS (8)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# NEW: 7 SEQUENCE-BASED FEATURE HELPERS
# ─────────────────────────────────────────────

def get_sentence_length_jump(text):
    """Maximum jump between consecutive sentence lengths. Humans = big jumps, AI = uniform."""
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.5  # neutral baseline for short text
    jumps = [abs(lengths[i + 1] - lengths[i]) for i in range(len(lengths) - 1)]
    return float(max(jumps)) if jumps else 0.5


def get_topic_change_score(text):
    """Vocabulary overlap between thirds. Low overlap = topic changed = more human."""
    words = text.lower().split()
    if len(words) < 15:
        return 0.5  # neutral baseline for short text
    third = len(words) // 3
    p1 = set(words[:third])
    p2 = set(words[third:2 * third])
    p3 = set(words[2 * third:])
    overlap_12 = len(p1 & p2) / max(len(p1 | p2), 1)
    overlap_23 = len(p2 & p3) / max(len(p2 | p3), 1)
    return 1.0 - ((overlap_12 + overlap_23) / 2)


def get_thought_interruption_score(text):
    """How often the writer interrupts their own thought."""
    interruptions = [
        text.lower().count('wait'),
        text.lower().count('actually'),
        text.lower().count('no wait'),
        text.lower().count('ok so'),
        text.lower().count('anyway'),
        text.lower().count('but also'),
        text.lower().count('oh also'),
        text.lower().count('also btw'),
        len(re.findall(r'\.{3}', text)),
        len(re.findall(r'\-{1,2}', text)),
    ]
    return float(sum(interruptions))


def get_logical_flow_score(text):
    """AI text uses explicit logical connectors; humans have abrupt transitions."""
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
        len(re.findall(r'\b\d+\b', text)),
        len(re.findall(r'\b[A-Z][a-z]+\b', text)),
        text.lower().count('my '),
        text.lower().count('i '),
        len(re.findall(
            r'yesterday|today|tomorrow|last week|this morning|tonight|right now|rn',
            text.lower()
        )),
    ]
    return float(sum(specific_patterns))


def get_emotional_variance(text):
    """High variance in emotional intensity between text halves = human."""
    emotional_words = [
        '!', '?', 'omg', 'lol', 'lmao', 'wtf',
        'hate', 'love', 'scared', 'excited',
        'ugh', 'yay', 'noooo', 'omgggg', 'pls',
        'pleaseee', 'dying', 'dead', 'crying',
    ]
    words = text.lower().split()
    if len(words) < 10:
        return 0.5  # neutral baseline for short text
    mid = len(words) // 2
    first_half = ' '.join(words[:mid])
    second_half = ' '.join(words[mid:])
    score1 = sum(first_half.count(e) for e in emotional_words)
    score2 = sum(second_half.count(e) for e in emotional_words)
    return float(abs(score1 - score2))


def get_sentence_starter_randomness(text):
    """Humans start sentences with casual words repetitively; AI uses structured variety."""
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


def get_structural_uniformity(text):
    """HIGH value = uniform sentence lengths = AI-like. LOW = irregular = human."""
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 3:
        return 0.5
    variance = float(np.std(lengths))
    return 1.0 / (variance + 1.0)


# ─────────────────────────────────────────────
# FEATURE EXTRACTION (28 features, same order as train_model.py)
# ─────────────────────────────────────────────

def extract_features(text):
    words = text.split()
    n = len(words) or 1

    # Original 5
    slang    = sum(1 for w in words if w.lower() in ["lol", "ngl", "idk", "bruh"])
    exclam   = text.count("!")
    ques     = text.count("?")
    avg_len  = np.mean([len(w) for w in words]) if words else 0
    unique   = len(set(words)) / n

    # Original 8 burstiness/style
    burst    = get_burstiness(text)
    perp     = get_perplexity_proxy(text)
    drift    = get_topic_drift(text)
    filler   = get_filler_count(text)
    selfcorr = get_self_correction(text)
    ss_var   = get_sentence_start_variety(text)
    punc_var = get_punctuation_variance(text)
    hedge    = get_hedge_ratio(text)

    # Original 8 stylometric
    ttr      = get_type_token_ratio(text)
    sent_cmp = get_avg_sentence_complexity(text)
    disc     = get_discourse_markers(text)
    q_auth   = get_question_authenticity(text)
    # Reduce slang-proxy weights: scale down emotional_authenticity and internet_language
    emo      = get_emotional_authenticity(text) * 0.4
    narr     = get_narrative_inconsistency(text)
    inet     = get_internet_language_score(text) * 0.4
    gram     = get_grammar_intentionality(text)

    # New 7 sequence-based — structural features up-weighted
    sl_jump  = get_sentence_length_jump(text) * 1.5
    tc_score = get_topic_change_score(text) * 1.5
    thought  = get_thought_interruption_score(text)
    logic    = get_logical_flow_score(text)
    personal = get_personal_specificity(text)
    emo_var  = get_emotional_variance(text)
    starter  = get_sentence_starter_randomness(text)

    return np.array([[
        slang, exclam, ques, avg_len, unique,
        burst, perp, drift, filler, selfcorr, ss_var, punc_var, hedge,
        ttr, sent_cmp, disc, q_auth, emo, narr, inet, gram,
        sl_jump, tc_score, thought, logic, personal, emo_var, starter,
    ]])


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


# ─────────────────────────────────────────────
# AI-HUMANIZED OVERRIDE (uses new sequence features)
# ─────────────────────────────────────────────

def ai_humanized_override(feat_dict, prediction, confidence):
    """
    Rule-based post-processing using structural features.
    Returns (prediction, confidence, detection_type, ai_humanized_risk).
    """
    word_count = feat_dict.get('word_count', 0)

    # HARD RULE: very short text cannot contain AI structural signals
    if word_count <= 4:
        return 0, max(confidence, 70.0), 'Human', 'LOW'

    struct_unif = feat_dict.get('structural_uniformity', 0.5)
    topic_drift  = feat_dict.get('topic_drift', 0.5)
    sl_jump      = feat_dict.get('sentence_length_jump', 0.5)

    # Compute AI-humanized risk score from structural signals
    risk_score = (struct_unif + topic_drift) / 2.0
    if risk_score > 0.75:
        ai_humanized_risk = 'HIGH'
    elif risk_score > 0.5:
        ai_humanized_risk = 'MEDIUM'
    else:
        ai_humanized_risk = 'LOW'

    # Primary AI-humanized detection: model said AI + strong structural uniformity
    if (prediction == 1
            and struct_unif > 0.7
            and topic_drift > 0.6
            and sl_jump < 0.4):
        new_conf = min(confidence + 0.15, 90.0)
        return 1, new_conf, 'AI-Humanized', ai_humanized_risk

    # Override: model said Human but structural signals are very AI-like
    if (prediction == 0
            and struct_unif > 0.75
            and topic_drift > 0.7):
        return 1, min(confidence + 0.10, 85.0), 'AI-Humanized', 'HIGH'

    detection_type = 'Normal'
    return prediction, confidence, detection_type, ai_humanized_risk


# ─────────────────────────────────────────────
# DECISION LOGIC
# ─────────────────────────────────────────────

def get_decision(confidence, prediction=None):
    # Support legacy callers that pass only confidence as a percentage (0-100)
    # and new callers that pass prediction separately
    if prediction is None:
        # Legacy path: treat confidence as a 0-100 percentage
        if confidence >= 75:
            return {"label": "Acceptable", "color": "green", "icon": "checkmark", "message": "High confidence"}
        elif confidence >= 55:
            return {"label": "Needs Review", "color": "yellow", "icon": "warning", "message": "Moderate confidence"}
        else:
            return {"label": "Likely AI", "color": "red", "icon": "cross", "message": "Low confidence"}

    # New path: prediction-aware thresholds
    # confidence here is a 0-100 percentage
    if prediction == 0:  # Human prediction — be generous
        if confidence >= 60:
            return {
                "label":   "Acceptable",
                "icon":    "checkmark",
                "color":   "green",
                "message": "High certainty — Human written text.",
            }
        elif confidence >= 45:
            return {
                "label":   "Likely Human",
                "icon":    "flag",
                "color":   "yellow",
                "message": "Moderate certainty — Likely human written.",
            }
        else:
            return {
                "label":   "Needs Review",
                "icon":    "warning",
                "color":   "red",
                "message": "Low confidence — verify manually.",
            }
    else:  # AI prediction — be strict
        if confidence >= 75:
            return {
                "label":   "AI Generated",
                "icon":    "robot",
                "color":   "red",
                "message": "High certainty — AI generated text detected.",
            }
        elif confidence >= 55:
            return {
                "label":   "Likely AI",
                "icon":    "warning",
                "color":   "yellow",
                "message": "Moderate certainty — likely AI generated.",
            }
        else:
            return {
                "label":   "Needs Review",
                "icon":    "question",
                "color":   "yellow",
                "message": "Borderline — could be human or AI.",
            }


# ─────────────────────────────────────────────
# SAFE MODEL RUNNER
# ─────────────────────────────────────────────

def run_model(model, X, threshold=0.5):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
        else:
            pred = model.predict(X)[0]
            proba = [0.5, 0.5]
            proba[pred] = 1.0

        pred = int(proba[1] >= threshold)
        conf = round(max(proba) * 100, 2)

        return {
            "prediction":  pred,
            "confidence":  conf,
            "proba_human": round(proba[0] * 100, 2),
            "proba_ai":    round(proba[1] * 100, 2),
            "label":       "Human" if pred == 0 else "AI"
        }

    except Exception as e:
        print("MODEL ERROR:", e)
        return {
            "prediction":  0,
            "confidence":  0,
            "proba_human": 0,
            "proba_ai":    0,
            "label":       "Error"
        }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/stats")
def stats():
    try:
        s = joblib.load("model/stats.pkl")
        return jsonify(s)
    except Exception as e:
        print("Stats error:", e)
        return jsonify({
            "lr_accuracy":       0,
            "nb_accuracy":       0,
            "rf_accuracy":       0,
            "svm_accuracy":      0,
            "ensemble_accuracy": 0,
            "best_threshold":    0.5,
        })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if not data or "text" not in data:
            return jsonify({"error": "No input text"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400

        print("INPUT:", text[:80])

        # PREPROCESS
        clean = preprocess(text)

        # TF-IDF + CHAR VECTORIZER
        tfidf_x = tfidf_vec.transform([clean])
        char_x  = char_vec.transform([clean])

        # FEATURES (use original text for punctuation/slang detection)
        feat_raw    = extract_features(text)
        feat_scaled = scaler.transform(feat_raw)

        # COMBINE: tfidf + char + hand-crafted features
        combo    = hstack([tfidf_x, char_x, csr_matrix(feat_scaled)])
        combo_nb = hstack([tfidf_x, char_x, csr_matrix(feat_raw)])

        # RUN MODELS
        res_lr  = run_model(lr,  combo)
        res_rf  = run_model(rf,  combo)
        res_svm = run_model(svm, combo)
        res_nb  = run_model(nb,  combo_nb)
        res_ens = run_model(ensemble, combo, threshold=BEST_THRESHOLD)

        final = res_ens

        # Compute all named feature values
        words    = text.split()
        burst    = round(get_burstiness(text), 3)
        perp     = round(get_perplexity_proxy(text), 3)
        drift    = round(get_topic_drift(text), 3)
        filler   = int(get_filler_count(text))
        selfcorr = int(get_self_correction(text))
        ss_var   = round(get_sentence_start_variety(text), 3)
        punc_var = round(get_punctuation_variance(text), 3)
        hedge    = round(get_hedge_ratio(text), 4)
        ttr      = round(get_type_token_ratio(text), 3)
        sent_cmp = round(get_avg_sentence_complexity(text), 3)
        disc     = int(get_discourse_markers(text))
        q_auth   = round(get_question_authenticity(text), 3)
        emo      = int(get_emotional_authenticity(text))
        narr     = int(get_narrative_inconsistency(text))
        inet     = round(get_internet_language_score(text), 4)
        gram     = round(get_grammar_intentionality(text), 2)

        # New sequence features
        sl_jump  = round(get_sentence_length_jump(text), 2)
        tc_score = round(get_topic_change_score(text), 3)
        thought  = round(get_thought_interruption_score(text), 2)
        logic    = round(get_logical_flow_score(text), 3)
        personal = round(get_personal_specificity(text), 2)
        emo_var  = round(get_emotional_variance(text), 2)
        starter  = round(get_sentence_starter_randomness(text), 3)
        struct_u = round(get_structural_uniformity(text), 4)

        feat_dict = {
            "word_count":               len(words),
            "avg_word_length":          round(np.mean([len(w) for w in words]) if words else 0, 2),
            "punctuation":              sum(1 for c in text if c in ".,!?"),
            "burstiness":               burst,
            "perplexity_proxy":         perp,
            "topic_drift":              drift,
            "filler_words":             filler,
            "self_corrections":         selfcorr,
            "casual_starters":          ss_var,
            "punct_variance":           punc_var,
            "hedge_ratio":              hedge,
            "type_token_ratio":         ttr,
            "sentence_complexity":      sent_cmp,
            "discourse_markers":        disc,
            "question_authenticity":    q_auth,
            "emotional_authenticity":   emo,
            "narrative_inconsistency":  narr,
            "internet_language":        inet,
            "grammar_intentionality":   gram,
            # New sequence features
            "sentence_length_jump":     sl_jump,
            "topic_change_score":       tc_score,
            "thought_interruption":     thought,
            "logical_flow_score":       logic,
            "personal_specificity":     personal,
            "emotional_variance":       emo_var,
            "sentence_starter_randomness": starter,
            "structural_uniformity":    struct_u,
        }

        # POST-PROCESSING OVERRIDE
        final_pred, final_conf, detection_type, ai_humanized_risk = ai_humanized_override(
            feat_dict, final["prediction"], final["confidence"]
        )

        final_label = "Human" if final_pred == 0 else "AI"

        return jsonify({
            "prediction":         final_pred,
            "label":              final_label,
            "confidence":         final_conf,
            "proba_human":        final["proba_human"],
            "proba_ai":           final["proba_ai"],
            "decision":           get_decision(final_conf, final_pred),
            "model_used":         "Ensemble",
            "detection_type":     detection_type,
            "ai_humanized_risk":  ai_humanized_risk,

            "features": feat_dict,

            "all_models": [
                {"model_name": "LR",       **res_lr},
                {"model_name": "NB",       **res_nb},
                {"model_name": "RF",       **res_rf},
                {"model_name": "SVM",      **res_svm},
                {"model_name": "ENSEMBLE", **res_ens},
            ]
        })

    except Exception as e:
        print("BACKEND ERROR:", e)
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
