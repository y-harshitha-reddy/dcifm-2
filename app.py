
import io
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

APP_NAME = "DCIFM"
APP_SUBTITLE = "Dynamic Consumer Identity Forecasting Model"
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "DCIFM_Master_Consumer_Dataset_10000.csv"
MODEL_DIR = BASE / "models"
DB_PATH = BASE / "dcifm.db"

st.set_page_config(page_title="DCIFM", page_icon="🧠", layout="wide")

DOMAINS = ["Fitness","Wellness","Technology","Fashion","Travel",
           "Learning","Entertainment","Food","Home","Finance"]

IDENTITIES = [
    "Health-Conscious Lifestyle Adopter",
    "Technology-Oriented Adopter",
    "Learning-Oriented Explorer",
    "Travel & Adventure Explorer",
    "Fashion & Lifestyle Explorer",
    "Home & Lifestyle Optimizer",
    "Food & Wellness Explorer",
    "Entertainment & Social Experience Seeker",
    "Finance-Aware Planner",
    "Balanced Multi-Interest Consumer"
]

IDENTITY_DOMAINS = {
    "Health-Conscious Lifestyle Adopter": ["Wellness","Fitness","Food"],
    "Technology-Oriented Adopter": ["Technology"],
    "Learning-Oriented Explorer": ["Learning","Technology"],
    "Travel & Adventure Explorer": ["Travel"],
    "Fashion & Lifestyle Explorer": ["Fashion"],
    "Home & Lifestyle Optimizer": ["Home","Technology"],
    "Food & Wellness Explorer": ["Food","Wellness"],
    "Entertainment & Social Experience Seeker": ["Entertainment","Fashion"],
    "Finance-Aware Planner": ["Finance","Technology"],
    "Balanced Multi-Interest Consumer": DOMAINS
}

FUTURE_NEEDS = {
    "Fitness": [("Wearable fitness tracker","activity monitoring"),
                ("Personalized nutrition program","health and fitness goals"),
                ("Fitness coaching or membership","continued exercise"),
                ("Sports recovery products","training recovery")],
    "Wellness": [("Wellness subscription","sustained healthy routines"),
                 ("Sleep or meditation app","wellbeing routines"),
                 ("Health monitoring device","personal tracking"),
                 ("Nutrition planning service","healthier habits")],
    "Technology": [("Smart device or accessory","technology adoption"),
                   ("Productivity software","digital workflow"),
                   ("Online technology course","skill development"),
                   ("Cloud or digital service","connected usage")],
    "Fashion": [("Seasonal apparel","wardrobe development"),
                ("Accessories","style expression"),
                ("Personal styling service","style discovery"),
                ("Beauty or grooming products","lifestyle presentation")],
    "Travel": [("Travel luggage","travel preparation"),
               ("Travel insurance","trip planning"),
               ("Destination experience","travel intent"),
               ("Accommodation or tour","trip execution")],
    "Learning": [("Professional certification","skill development"),
                 ("Online course","continued learning"),
                 ("Learning subscription","recurring education"),
                 ("Career-oriented workshop","applied learning")],
    "Entertainment": [("Streaming subscription","continued content use"),
                      ("Event or concert","experience engagement"),
                      ("Gaming accessory","digital entertainment"),
                      ("Creator/community experience","social entertainment")],
    "Food": [("Healthy meal subscription","food and lifestyle interests"),
             ("Specialty food products","food exploration"),
             ("Cooking course","food-related learning"),
             ("Grocery subscription","recurring consumption")],
    "Home": [("Smart home device","home technology"),
             ("Home organization products","home optimization"),
             ("Kitchen appliance","household routines"),
             ("Home decor","lifestyle expression")],
    "Finance": [("Personal finance tool","financial planning"),
                ("Investment education","financial learning"),
                ("Budgeting service","spending control"),
                ("Insurance product","risk management")]
}

# COCO labels are useful but limited. YOLO-World is preferred because it
# can use a custom vocabulary relevant to consumer context.
VISION_CLASSES = [
    "smartphone", "cell phone", "laptop", "tablet", "headphones",
    "camera", "smartwatch", "watch", "bicycle", "sports equipment",
    "running shoe", "sneaker", "shoe", "backpack", "suitcase",
    "yoga mat", "dumbbell", "exercise equipment", "water bottle",
    "protein powder container", "book", "keyboard", "mouse",
    "television", "sofa", "chair", "dining table", "bottle"
]

VISION_DOMAIN_MAP = {
    "bicycle": {"Fitness": .90, "Travel": .25},
    "sports equipment": {"Fitness": .95},
    "running shoe": {"Fitness": .90, "Fashion": .35},
    "sneaker": {"Fitness": .55, "Fashion": .70},
    "shoe": {"Fitness": .35, "Fashion": .70},
    "yoga mat": {"Fitness": .90, "Wellness": .65},
    "dumbbell": {"Fitness": .95},
    "exercise equipment": {"Fitness": .95},
    "water bottle": {"Fitness": .30, "Wellness": .40},
    "protein powder container": {"Fitness": .55, "Wellness": .80, "Food": .35},
    "smartphone": {"Technology": .95},
    "cell phone": {"Technology": .95},
    "laptop": {"Technology": .90, "Learning": .25},
    "tablet": {"Technology": .80, "Learning": .30},
    "headphones": {"Technology": .75, "Entertainment": .45},
    "camera": {"Technology": .80, "Entertainment": .30},
    "smartwatch": {"Technology": .75, "Fitness": .65},
    "watch": {"Technology": .25, "Fashion": .35},
    "backpack": {"Travel": .55, "Learning": .30},
    "suitcase": {"Travel": .95},
    "book": {"Learning": .95},
    "keyboard": {"Technology": .80, "Learning": .25},
    "mouse": {"Technology": .75},
    "television": {"Entertainment": .85, "Technology": .25},
    "sofa": {"Home": .85},
    "chair": {"Home": .55},
    "dining table": {"Home": .75},
    "bottle": {"Food": .25, "Wellness": .20},
}

GROUND_TRUTH_COLS = [
    "identity_orientation_ground_truth",
    "identity_strength_ground_truth",
    "emerging_identity_ground_truth",
    "identity_change_score_ground_truth",
    "future_need_category_ground_truth",
    "future_purchase_probability_ground_truth",
    "future_purchase_30d_ground_truth",
    "future_purchase_value_ground_truth",
]

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["consumer_id"] = df["consumer_id"].astype(str)
    return df

def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = db_conn()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        password_hash TEXT,
        role TEXT
    );
    CREATE TABLE IF NOT EXISTS audit_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT,
        timestamp TEXT,
        details TEXT
    );
    """)
    demo = [
        ("executive","executive123","Executive"),
        ("analyst","analyst123","Analyst"),
        ("researcher","researcher123","Researcher"),
        ("user","user123","User"),
    ]
    for u,p,r in demo:
        con.execute("INSERT OR IGNORE INTO users VALUES(?,?,?)",
                    (u, hashlib.sha256(p.encode()).hexdigest(), r))
    con.commit()
    con.close()

def audit(action, details=""):
    try:
        u = st.session_state.get("auth", {}).get("username","unknown")
        con = db_conn()
        con.execute("INSERT INTO audit_logs(username,action,timestamp,details) VALUES(?,?,?,?)",
                    (u, action, datetime.now().isoformat(), str(details)))
        con.commit()
        con.close()
    except Exception:
        pass

def login(username, password):
    con = db_conn()
    row = con.execute("SELECT username,password_hash,role FROM users WHERE username=?",
                      (username.strip(),)).fetchone()
    con.close()
    if row and hashlib.sha256(password.encode()).hexdigest() == row[1]:
        return {"username": row[0], "role": row[2]}
    return None

def require_login():
    if "auth" not in st.session_state:
        st.session_state.auth = None
    if st.session_state.auth is None:
        st.title("🧠 DCIFM")
        st.caption(APP_SUBTITLE)
        st.markdown("### Research Prototype Login")
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            ok = st.form_submit_button("Login", use_container_width=True)
        if ok:
            user = login(u,p)
            if user:
                st.session_state.auth = user
                audit("login")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.info("Demo: executive/executive123 · analyst/analyst123 · researcher/researcher123 · user/user123")
        st.stop()

def feature_columns(df):
    # Explicitly exclude identifiers and all evaluation-only targets.
    cols = [c for c in df.select_dtypes(include=np.number).columns
            if c not in GROUND_TRUTH_COLS]
    return cols

def domain_cols(df):
    return [f"{d.lower()}_score" for d in DOMAINS if f"{d.lower()}_score" in df.columns]

def current_identity_from_domains(row):
    scores = {d: float(row.get(f"{d.lower()}_score",0)) for d in DOMAINS}
    vals = np.array(list(scores.values()), dtype=float)
    if vals.max() > 0:
        scores = {d:v/vals.max() for d,v in scores.items()}
    ranked = []
    for identity, ds in IDENTITY_DOMAINS.items():
        s = float(np.mean([scores.get(d,0) for d in ds]))
        ranked.append((identity,s))
    ranked.sort(key=lambda x:x[1], reverse=True)
    return ranked

@st.cache_resource
def train_identity_models(df):
    X = df[feature_columns(df)].replace([np.inf,-np.inf],np.nan).fillna(0)
    y_current = df["identity_orientation_ground_truth"].astype(str)
    y_emerging = df["emerging_identity_ground_truth"].astype(str)

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=.25, random_state=42, stratify=y_emerging
    )

    model_current = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ))
    ])
    model_emerging = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=350, max_depth=16, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=43, n_jobs=-1
        ))
    ])

    model_current.fit(X.iloc[idx_train], y_current.iloc[idx_train])
    model_emerging.fit(X.iloc[idx_train], y_emerging.iloc[idx_train])

    p_cur = model_current.predict_proba(X.iloc[idx_test])
    p_em = model_emerging.predict_proba(X.iloc[idx_test])

    current_pred = model_current.predict(X.iloc[idx_test])
    emerging_pred = model_emerging.predict(X.iloc[idx_test])

    metrics = {
        "current_accuracy": accuracy_score(y_current.iloc[idx_test], current_pred),
        "current_f1": f1_score(y_current.iloc[idx_test], current_pred, average="weighted"),
        "emerging_accuracy": accuracy_score(y_emerging.iloc[idx_test], emerging_pred),
        "emerging_f1": f1_score(y_emerging.iloc[idx_test], emerging_pred, average="weighted"),
        "test_size": len(idx_test),
        "idx_test": idx_test,
    }

    return model_current, model_emerging, metrics

def generate_predictions(df):
    cur_model, em_model, metrics = train_identity_models(df)
    X = df[feature_columns(df)].replace([np.inf,-np.inf],np.nan).fillna(0)

    cur_prob = cur_model.predict_proba(X)
    em_prob = em_model.predict_proba(X)
    cur_classes = cur_model.classes_
    em_classes = em_model.classes_

    current = cur_classes[cur_prob.argmax(axis=1)]
    emerging = em_classes[em_prob.argmax(axis=1)]
    current_conf = cur_prob.max(axis=1)
    emerging_conf = em_prob.max(axis=1)

    # Identity transition strength is the probability that the emerging
    # identity differs from the inferred current identity, adjusted by
    # confidence. This is not a psychological diagnosis.
    shift = current != emerging
    shift_strength = np.where(
        shift,
        np.clip((emerging_conf + current_conf)/2,0,1),
        0
    )

    rows = []
    for i, uid in enumerate(df["consumer_id"]):
        ranked = current_identity_from_domains(df.iloc[i])
        current_domain = ranked[0][0]
        current_domain_score = ranked[0][1]
        emerging_name = emerging[i]
        emerging_domains = IDENTITY_DOMAINS.get(emerging_name, ["Technology"])
        needs_domain = emerging_domains[0]
        needs = FUTURE_NEEDS.get(needs_domain, FUTURE_NEEDS["Technology"])

        rows.append({
            "consumer_id": uid,
            "current_identity": current[i],
            "emerging_identity": emerging[i],
            "current_confidence": float(current_conf[i]),
            "emerging_confidence": float(emerging_conf[i]),
            "identity_shift_detected": bool(shift[i]),
            "identity_shift_strength": float(shift_strength[i]),
            "dominant_domain": current_domain,
            "domain_alignment": float(current_domain_score),
            "future_need_domain": needs_domain,
            "future_need_1": needs[0][0],
            "future_need_2": needs[1][0],
            "future_need_3": needs[2][0],
        })

    pred = pd.DataFrame(rows).set_index("consumer_id")
    return pred, metrics

def get_state():
    if "df" not in st.session_state:
        st.session_state.df = load_dataset()
    if "pred" not in st.session_state:
        with st.spinner("Training DCIFM identity models on the synthetic research dataset..."):
            st.session_state.pred, st.session_state.metrics = generate_predictions(st.session_state.df)

def section_intro(title, text):
    st.title(title)
    st.info(text)

def page_dashboard():
    section_intro(
        "🧠 DCIFM Dashboard",
        "Overview of the full pipeline. The dashboard connects observed consumer behavior to inferred current identity, emerging identity, identity shifts, and anticipatory future-need categories. Charts summarize what the model sees and how consumers are distributed."
    )
    df, p = st.session_state.df, st.session_state.pred
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Consumers", f"{len(df):,}")
    c2.metric("Behavioral features", len(feature_columns(df)))
    c3.metric("Detected identity shifts", f"{p.identity_shift_detected.mean()*100:.1f}%")
    c4.metric("Emerging forecast confidence", f"{p.emerging_confidence.mean()*100:.1f}%")

    st.markdown("### What this section shows")
    st.caption("The first row gives the population scale and core model outputs. The identity distribution shows which emerging identity the model forecasts most often. The shift chart separates consumers whose forecast changes from those with no material identity transition.")

    counts = p["emerging_identity"].value_counts().reset_index()
    counts.columns = ["identity","consumers"]
    st.plotly_chart(px.bar(counts, x="identity", y="consumers",
                           title="Forecasted Emerging Identity Distribution"),
                    use_container_width=True)

    shift_counts = p["identity_shift_detected"].map({True:"Shift detected",False:"No material shift"}).value_counts().reset_index()
    shift_counts.columns = ["status","consumers"]
    st.plotly_chart(px.pie(shift_counts, names="status", values="consumers",
                           title="Identity Transition Status"),
                    use_container_width=True)

def page_profiles():
    section_intro(
        "👤 Consumer Profiles",
        "This section gives an individual consumer a deeper identity-level profile. It deliberately separates current identity from emerging identity and displays the behavioral evidence behind the forecast rather than simply repeating the consumer's past purchases."
    )
    df,p = st.session_state.df, st.session_state.pred
    shifted = p.index[p.identity_shift_detected].tolist()
    options = shifted if shifted else p.index.tolist()
    uid = st.selectbox("Select consumer (shifted consumers are prioritized)", options)
    row = p.loc[uid]
    raw = df.set_index("consumer_id").loc[uid]

    a,b,c,d = st.columns(4)
    a.metric("Current identity", row.current_identity)
    b.metric("Emerging identity", row.emerging_identity)
    c.metric("Shift", "Detected" if row.identity_shift_detected else "No material shift")
    d.metric("Forecast confidence", f"{row.emerging_confidence*100:.1f}%")

    if row.identity_shift_detected:
        st.success(f"Identity direction: {row.current_identity}  →  {row.emerging_identity}")
    else:
        st.warning("No material identity transition was detected for this consumer. The model should not manufacture a shift when the behavioral evidence does not support one.")

    st.markdown("### Behavioral domain profile")
    st.caption("The bars show normalized domain evidence derived from the consumer's observed behavioral features. They indicate relative behavioral orientation, not psychological traits.")
    vals = []
    for d in DOMAINS:
        vals.append({"domain":d, "score":float(raw.get(f"{d.lower()}_score",0))/100})
    chart = pd.DataFrame(vals).sort_values("score")
    st.plotly_chart(px.bar(chart,x="score",y="domain",orientation="h",range_x=[0,1],
                           title="Relative Behavioral-Domain Signals"),
                    use_container_width=True)

    st.markdown("### Why the forecast moved")
    st.caption("The model combines multiple signals such as searches, product views, engagement, purchases, category diversity, commercial intent and cross-category interactions. The table below highlights the strongest observed signals for this consumer.")
    evidence = pd.DataFrame({
        "signal":[
            "Search frequency/day","Engagement frequency/day","Commercial intent",
            "Category diversity","Cross-category interactions","Product diversity",
            "Recency (days)"
        ],
        "value":[
            raw.get("search_frequency_per_day",0),
            raw.get("engagement_frequency_per_day",0),
            raw.get("commercial_intent_score",0),
            raw.get("category_diversity",0),
            raw.get("cross_category_interactions",0),
            raw.get("product_diversity",0),
            raw.get("recency_days",0)
        ]
    })
    st.dataframe(evidence, use_container_width=True, hide_index=True)

    st.markdown("### Anticipatory future needs")
    st.caption("These are potential category-aligned needs inferred from the emerging identity forecast. They are not guaranteed purchases.")
    needs_domain = row.future_need_domain
    needs = FUTURE_NEEDS[needs_domain]
    st.dataframe(pd.DataFrame([{"Potential need":n,"Why it aligns":r} for n,r in needs]),
                 use_container_width=True, hide_index=True)

def page_behavior():
    section_intro(
        "📊 Behavioral Analytics",
        "This section converts the master consumer dataset into interpretable behavioral evidence. Use the selector to inspect distributions of searches, purchases, engagement, spending, intent and other signals that feed the identity models."
    )
    df = st.session_state.df
    num = feature_columns(df)
    col = st.selectbox("Behavioral feature", num)
    st.caption(f"This graph shows how the selected behavioral signal is distributed across the {len(df):,} consumers.")
    st.plotly_chart(px.histogram(df,x=col,nbins=40,title=f"Distribution of {col}"),
                    use_container_width=True)

    summary_cols = ["total_interactions","searches","product_views","purchases",
                    "total_spend","category_diversity","commercial_intent_score"]
    st.markdown("### Population behavioral summary")
    st.caption("These summary statistics describe the scale and variation of the behavioral inputs before identity inference.")
    st.dataframe(df[summary_cols].describe().T.round(2), use_container_width=True)

def page_segmentation():
    section_intro(
        "🧩 Behavioral Segmentation",
        "Clustering groups consumers by similarity in observed behavior. It is a contextual layer: segments help analysts see recurring behavioral patterns, while DCIFM identity forecasting separately estimates current and emerging identity orientations."
    )
    df = st.session_state.df
    X = df[feature_columns(df)].replace([np.inf,-np.inf],np.nan).fillna(0)
    k = st.slider("Number of behavioral segments",2,8,5)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k,n_init=20,random_state=42)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs,labels)
    dbi = davies_bouldin_score(Xs,labels)

    a,b = st.columns(2)
    a.metric("Silhouette score",f"{sil:.3f}")
    b.metric("Davies-Bouldin index",f"{dbi:.3f}")
    st.caption("Higher Silhouette and lower Davies-Bouldin generally indicate cleaner separation; these are segmentation diagnostics, not identity-forecast accuracy metrics.")

    xy = PCA(n_components=2,random_state=42).fit_transform(Xs)
    plot = pd.DataFrame({"PC1":xy[:,0],"PC2":xy[:,1],"segment":labels.astype(str)})
    st.plotly_chart(px.scatter(plot,x="PC1",y="PC2",color="segment",
                               title="Behavioral Segment Map"),
                    use_container_width=True)

def page_identity():
    section_intro(
        "🧬 Identity Analysis",
        "This section compares the model's inferred current identity with its independently forecast emerging identity. A transition is reported only when the two model outputs differ; the interface does not force every consumer to have an identity shift."
    )
    p = st.session_state.pred
    table = p[["current_identity","emerging_identity","identity_shift_detected",
               "current_confidence","emerging_confidence","identity_shift_strength"]].copy()
    table["current_confidence"]=(table.current_confidence*100).round(1)
    table["emerging_confidence"]=(table.emerging_confidence*100).round(1)
    table.columns=["Current identity","Emerging identity","Shift detected",
                   "Current confidence %","Emerging confidence %","Shift strength"]
    st.dataframe(table.head(200),use_container_width=True)

    shift = p[p.identity_shift_detected]
    st.markdown("### Identity transition matrix")
    st.caption("This matrix counts consumers moving from each inferred current identity to each forecast emerging identity. Off-diagonal cells represent identity transitions.")
    matrix = pd.crosstab(p.current_identity,p.emerging_identity)
    st.plotly_chart(px.imshow(matrix,text_auto=True,aspect="auto",
                               title="Current → Emerging Identity Transitions"),
                    use_container_width=True)

def page_forecast():
    section_intro(
        "🔮 Identity Forecast & Future Need",
        "This is the core DCIFM forecasting layer. It compares the model's current identity inference with a separately trained emerging-identity predictor. When a transition is detected, the model maps the emerging identity to plausible future need categories and displays why those needs are relevant."
    )
    p = st.session_state.pred
    shifted = p.index[p.identity_shift_detected].tolist()
    options = shifted if shifted else p.index.tolist()
    uid = st.selectbox("Consumer",options)
    r = p.loc[uid]

    st.markdown(f"### {r.current_identity}  →  **{r.emerging_identity}**")
    if r.identity_shift_detected:
        st.success(f"Identity shift detected · strength {r.identity_shift_strength*100:.1f}%")
    else:
        st.info("No material shift detected. The model is preserving the current identity rather than inventing a transition.")

    a,b,c = st.columns(3)
    a.metric("Current confidence",f"{r.current_confidence*100:.1f}%")
    b.metric("Emerging confidence",f"{r.emerging_confidence*100:.1f}%")
    c.metric("Forecast need domain",r.future_need_domain)

    st.markdown("### Future-need forecast")
    st.caption("The outputs below are anticipatory marketing hypotheses. They identify products/services that are semantically aligned with the forecasted identity direction; they are not guaranteed purchase predictions.")
    needs = FUTURE_NEEDS[r.future_need_domain]
    st.dataframe(pd.DataFrame([
        {"Rank":i+1,"Potential future need":n,"Marketing rationale":reason}
        for i,(n,reason) in enumerate(needs)
    ]),use_container_width=True,hide_index=True)

def load_yolo_world():
    if YOLO is None:
        return None, "Ultralytics is not installed."
    try:
        model = YOLO(str(MODEL_DIR/"yolov8s-worldv2.pt"))
        model.set_classes(VISION_CLASSES)
        return model, None
    except Exception as e:
        return None, f"YOLO-World could not be loaded: {e}"

def load_yolo_coco():
    if YOLO is None:
        return None
    try:
        return YOLO(str(MODEL_DIR/"yolov8n.pt"))
    except Exception:
        return None

@st.cache_resource
def cached_yolo_world():
    return load_yolo_world()

@st.cache_resource
def cached_yolo_coco():
    return load_yolo_coco()

def analyze_image(image_bytes, confidence=0.25):
    if Image is None:
        return None, "Pillow is unavailable."
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preferred: open-vocabulary YOLO-World with consumer-relevant classes.
    model, err = cached_yolo_world()
    mode = "YOLO-World"
    if model is None:
        model = cached_yolo_coco()
        mode = "YOLOv8n fallback"
    if model is None:
        return None, err or "No YOLO model is available."

    try:
        result = model.predict(source=img,conf=confidence,verbose=False)[0]
        names = result.names
        dets=[]
        if result.boxes is not None:
            for box in result.boxes:
                cls=int(box.cls.item())
                score=float(box.conf.item())
                xyxy=[round(float(v),1) for v in box.xyxy[0].tolist()]
                label=names.get(cls,str(cls))
                dets.append({"object":label,"confidence":round(score,3),"bbox":xyxy})

        annotated=result.plot()
        # Ultralytics returns BGR numpy arrays for plot().
        annotated=annotated[:,:,::-1]

        visual_scores={d:0.0 for d in DOMAINS}
        for d in dets:
            label=d["object"].lower()
            mapping=VISION_DOMAIN_MAP.get(label,{})
            for domain,weight in mapping.items():
                visual_scores[domain]+=weight*d["confidence"]

        if max(visual_scores.values())>0:
            mx=max(visual_scores.values())
            visual_scores={k:min(1,v/mx) for k,v in visual_scores.items()}

        return {"annotated":annotated,"detections":dets,
                "visual_scores":visual_scores,"model_mode":mode},None
    except Exception as e:
        return None,f"YOLO inference failed: {e}"

def page_vision():
    section_intro(
        "👁️ YOLO Visual Behavioral Intelligence",
        "This is a supporting evidence layer. YOLO does not identify a consumer or infer psychology. It detects objects visible in a consumer-related image, then DCIFM maps those objects to broad behavioral domains. This can add contextual evidence to an existing behavioral pattern—for example, a bicycle or yoga mat can support a fitness/wellness signal."
    )
    st.markdown("### Why YOLO is included")
    st.caption("The visual branch addresses a multimodal question: can observable visual context complement digital behavioral signals? It is useful when an image contains objects that are semantically related to consumer activity. It is not a replacement for behavioral data.")

    upload=st.file_uploader("Upload an image",type=["jpg","jpeg","png","webp"])
    conf=st.slider("Detection confidence threshold",0.10,0.90,0.20,0.05)
    if upload is None:
        st.info("Upload an image to run the visual branch.")
        return

    data=upload.getvalue()
    result,err=analyze_image(data,conf)
    if err:
        st.error(err)
        st.info("On first run, Ultralytics may download YOLO-World weights. An internet connection may be required.")
        return

    a,b=st.columns(2)
    with a:
        st.image(data,caption="Original image",use_container_width=True)
    with b:
        st.image(result["annotated"],caption=f"{result['model_mode']} detections",use_container_width=True)

    det_df=pd.DataFrame(result["detections"])
    st.markdown("### Detected objects")
    st.caption("Each row is a detected object, its confidence, and its bounding-box coordinates. Detection is separate from identity inference.")
    if det_df.empty:
        st.warning("No objects passed the selected confidence threshold. Lower the threshold slightly or use a clearer image.")
    else:
        st.dataframe(det_df,use_container_width=True,hide_index=True)

    vs=pd.DataFrame({"domain":list(result["visual_scores"].keys()),
                     "score":list(result["visual_scores"].values())})
    vs=vs.sort_values("score",ascending=False)
    st.markdown("### Visual context contribution")
    st.caption("These scores show how detected objects map into DCIFM's transparent behavioral ontology. They are contextual support signals, not consumer-identity scores by themselves.")
    st.plotly_chart(px.bar(vs.head(8),x="score",y="domain",orientation="h",
                           range_x=[0,1],title="Visual Context by Behavioral Domain"),
                    use_container_width=True)

    st.warning("Research limitation: YOLO-World/YOLOv8 pretrained weights are generic object detectors. They are not trained here on consumer-identity labels.")

def page_data():
    section_intro(
        "🗂️ Data Management",
        "This prototype is preconfigured for the unified DCIFM master dataset. The dataset uses one consumer_id per row and contains cross-source behavioral aggregates plus evaluation-only ground-truth labels. Ground-truth columns are excluded from model inputs to prevent target leakage."
    )
    df=st.session_state.df
    st.metric("Rows",f"{len(df):,}")
    st.metric("Columns",f"{len(df.columns)}")
    st.dataframe(df.head(30),use_container_width=True)

    st.download_button(
        "Download current dataset",
        data=df.to_csv(index=False).encode(),
        file_name="DCIFM_Master_Consumer_Dataset_10000.csv",
        mime="text/csv"
    )

    st.markdown("### Ground-truth separation")
    st.caption("The following columns are reserved for research evaluation and are never supplied to the identity models.")
    st.code("\n".join(GROUND_TRUTH_COLS))

def page_research():
    section_intro(
        "🧪 Research & Evaluation",
        "This section separates model validation from dashboard confidence. Confidence is the classifier's probability output; it is not the same as validated accuracy. The supplied synthetic dataset includes explicit ground-truth targets, allowing controlled evaluation."
    )
    m=st.session_state.metrics
    a,b,c,d=st.columns(4)
    a.metric("Current identity accuracy",f"{m['current_accuracy']*100:.1f}%")
    b.metric("Current identity F1",f"{m['current_f1']:.3f}")
    c.metric("Emerging identity accuracy",f"{m['emerging_accuracy']*100:.1f}%")
    d.metric("Emerging identity F1",f"{m['emerging_f1']:.3f}")

    st.caption(f"Metrics use a stratified 75/25 hold-out split of the synthetic dataset (test n={m['test_size']:,}). They demonstrate controlled prototype behavior, not real-world consumer accuracy.")

    p=st.session_state.pred
    st.markdown("### Identity transition validation")
    st.caption("This compares the model's inferred current/emerging identities with the synthetic labels used only for evaluation. It is useful for checking whether the transition mechanism is producing distinct directions.")
    eval_df=pd.DataFrame({
        "actual_current":st.session_state.df["identity_orientation_ground_truth"],
        "predicted_current":p["current_identity"],
        "actual_emerging":st.session_state.df["emerging_identity_ground_truth"],
        "predicted_emerging":p["emerging_identity"],
    })
    st.dataframe(eval_df.head(100),use_container_width=True)

    st.markdown("### Methodological safeguards")
    st.write(
        "• Ground-truth identity/future columns are excluded from model features.\n"
        "• Current and emerging identities are predicted by separate classifiers.\n"
        "• A transition is reported only when the two predicted identities differ.\n"
        "• Future-need recommendations are treated as anticipatory hypotheses, not guaranteed purchases.\n"
        "• Visual detection is a supporting modality and is not treated as psychological inference."
    )

def page_admin():
    section_intro(
        "🔐 Audit Logs",
        "This section records prototype actions for traceability. It is an application audit trail, not a tamper-proof production security ledger."
    )
    con=db_conn()
    logs=pd.read_sql_query("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 500",con)
    con.close()
    st.dataframe(logs,use_container_width=True)

def sidebar():
    st.sidebar.title("🧠 DCIFM")
    st.sidebar.caption("Dynamic Consumer Identity Forecasting Model")
    u=st.session_state.auth
    st.sidebar.write(f"**{u['username']}** · {u['role']}")
    if st.sidebar.button("Logout",use_container_width=True):
        audit("logout")
        st.session_state.auth=None
        st.rerun()
    st.sidebar.divider()
    pages=[
        ("Dashboard",page_dashboard),
        ("Consumer Profiles",page_profiles),
        ("Behavioral Analytics",page_behavior),
        ("Behavioral Segmentation",page_segmentation),
        ("Identity Analysis",page_identity),
        ("Identity Forecast",page_forecast),
        ("YOLO Visual Intelligence",page_vision),
        ("Data Management",page_data),
        ("Research & Evaluation",page_research),
    ]
    if u["role"] in {"Executive","Researcher"}:
        pages.append(("Audit Logs",page_admin))
    selected=st.sidebar.radio("Navigation",[x[0] for x in pages])
    return dict(pages)[selected]

init_db()
require_login()
get_state()
page=sidebar()
page()
