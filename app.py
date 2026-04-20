from pathlib import Path
import ast
import re
import warnings

import contractions
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Steam Price Intelligence System",
    page_icon="🎮",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(47, 92, 162, 0.20), transparent 28%),
            radial-gradient(circle at top left, rgba(0, 180, 216, 0.10), transparent 30%),
            linear-gradient(180deg, #0b1220 0%, #111827 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.76));
        box-shadow: 0 18px 45px rgba(2, 6, 23, 0.30);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        color: #f8fafc;
        font-size: 2rem;
    }
    .hero p {
        margin: 0;
        color: #cbd5e1;
        line-height: 1.55;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.70);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"
MODEL01_DIR = NOTEBOOKS_DIR / "model01_artifacts"
MODEL02_DIR = NOTEBOOKS_DIR / "model02_artifacts"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "games_march2025_fe.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "games_march2025_cleaned.csv"


EXAMPLES = {
    "Custom input": None,
    "Example 1 - Premium RPG": {
        "short_description": "An open world action RPG with deep combat mechanics, boss fights, crafting and immersive storytelling.",
        "required_age": 18,
        "release_year": 2026,
        "achievements": 60,
        "num_supported_languages": 14,
        "num_audio_languages": 8,
        "developers": ["FromSoftware"],
        "publishers": ["Bandai Namco"],
        "windows": 1,
        "mac": 0,
        "linux": 0,
        "genres": ["Action", "RPG"],
        "tags": ["Singleplayer", "Action", "Adventure", "3D"],
        "categories": ["Single-player", "Full controller support"],
    },
    "Example 2 - Battle Royale": {
        "short_description": "A fast-paced multiplayer battle royale shooter with skins, seasonal events and online competitive gameplay.",
        "required_age": 12,
        "release_year": 2025,
        "achievements": 5,
        "num_supported_languages": 10,
        "num_audio_languages": 3,
        "developers": ["Indie Studio"],
        "publishers": ["Self Published"],
        "windows": 1,
        "mac": 0,
        "linux": 0,
        "genres": ["Action"],
        "tags": ["Multiplayer", "Battle Royale", "Action"],
        "categories": ["Online PvP"],
    },
    "Example 3 - Cozy Simulation": {
        "short_description": "A relaxing pixel art farming simulator with crafting, exploration and cozy gameplay.",
        "required_age": 3,
        "release_year": 2026,
        "achievements": 12,
        "num_supported_languages": 6,
        "num_audio_languages": 1,
        "developers": ["Solo Dev"],
        "publishers": ["Solo Dev"],
        "windows": 1,
        "mac": 1,
        "linux": 0,
        "genres": ["Simulation", "Indie"],
        "tags": ["Singleplayer", "Casual", "2D"],
        "categories": ["Single-player"],
    },
}


def clean_description(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()

    leakage_words = [
        "free-to-play",
        "free to play",
        "purchases",
        "purchase",
        "free",
        "buy",
        "price",
        "in-app",
        "microtransaction",
    ]

    for word in leakage_words:
        text = text.replace(word, "")

    text = re.sub(r"[^\w\s]", "", text)
    text = contractions.fix(text)

    return text


def to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def format_category_label(col_name: str) -> str:
    text = col_name.replace("cat_", "").replace("_", " ")
    if text == "single player":
        return "Single-player"
    if text == "multi player":
        return "Multi-player"
    if text == "co op":
        return "Co-op"
    if text == "online co op":
        return "Online Co-op"
    if text == "local co op":
        return "Local Co-op"
    if text == "online pvp":
        return "Online PvP"
    if text == "shared split screen":
        return "Shared/Split Screen"
    if text == "full controller support":
        return "Full controller support"
    return text.title()


@st.cache_resource
def load_resources():
    stage1_model = joblib.load(MODEL01_DIR / "model_stage1.pkl")
    stage2_model = joblib.load(MODEL02_DIR / "model_stage2.pkl")
    scaler = joblib.load(MODEL01_DIR / "scaler.pkl")
    tfidf = joblib.load(MODEL01_DIR / "tfidf.pkl")
    threshold = float(joblib.load(MODEL01_DIR / "threshold.pkl"))
    feature_config = joblib.load(MODEL01_DIR / "feature_config.pkl")
    label_encoder = joblib.load(MODEL02_DIR / "labelencoder.pkl")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH, usecols=["developers", "publishers"])

    df["short_description_clean"] = df["short_description"].apply(clean_description)

    raw_df["developers"] = raw_df["developers"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    raw_df["publishers"] = raw_df["publishers"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    dev_counts = raw_df.explode("developers")["developers"].value_counts()
    publisher_counts = raw_df.explode("publishers")["publishers"].value_counts()

    dev_threshold = float(dev_counts.quantile(0.90))
    publisher_threshold = float(publisher_counts.quantile(0.90))

    reference_text_matrix = tfidf.transform(df["short_description_clean"])

    explanation_kb = {
        "genre_action": "Action-focused games in the dataset are more often monetized as paid products.",
        "genre_adventure": "Adventure games often support stronger paid pricing through story and content depth.",
        "genre_rpg": "RPG games usually indicate longer progression systems and stronger paid potential.",
        "genre_simulation": "Simulation titles often support paid pricing through replayability and content depth.",
        "genre_strategy": "Strategy titles frequently maintain paid pricing because of niche but committed audiences.",
        "tag_singleplayer": "Singleplayer tags are common in paid games across the training data.",
        "tag_action": "Action tags frequently appear in low, mid, and premium paid games.",
        "tag_adventure": "Adventure tags often align with content-heavy paid titles.",
        "tag_casual": "Casual tags can support lower price tiers when scope is smaller.",
        "tag_3d": "3D presentation is often associated with higher production value.",
        "tag_2d": "2D games appear across free and paid groups, but often align with budget and low price tiers.",
        "cat_single_player": "Single-player support reinforces a packaged paid game pattern.",
        "cat_multi_player": "Multiplayer support can increase scale and market reach.",
        "cat_full_controller_support": "Full controller support often appears in more polished releases.",
        "top_developer": "Top developers are more common in paid releases.",
        "top_publisher": "Top publishers usually support broader commercial launches.",
        "log_achievements": "A higher number of achievements suggests broader gameplay content.",
        "log_num_supported_languages": "More supported languages suggest wider target reach.",
        "log_num_audio_languages": "Audio localization often appears in higher-scope games.",
        "open world": "Open-world design often signals larger scope and stronger paid positioning.",
        "multiplayer": "Multiplayer can strengthen reach and retention, depending on the overall design.",
        "story": "Story-driven design often supports paid monetization.",
        "combat": "Combat-heavy design usually appears in stronger content-driven paid titles.",
        "battle royale": "Battle royale design is often linked with free-to-play or live-service patterns.",
        "live service": "Live-service updates often support recurring player retention.",
        "simulation": "Simulation elements often align with replayable paid games.",
        "crafting": "Crafting systems usually suggest more content depth and progression.",
    }

    return {
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "scaler": scaler,
        "tfidf": tfidf,
        "threshold": threshold,
        "feature_config": feature_config,
        "label_encoder": label_encoder,
        "df": df,
        "dev_counts": dev_counts,
        "publisher_counts": publisher_counts,
        "dev_threshold": dev_threshold,
        "publisher_threshold": publisher_threshold,
        "reference_text_matrix": reference_text_matrix,
        "explanation_kb": explanation_kb,
    }


resources = load_resources()

stage1_model = resources["stage1_model"]
stage2_model = resources["stage2_model"]
scaler = resources["scaler"]
tfidf = resources["tfidf"]
threshold = resources["threshold"]
feature_config = resources["feature_config"]
label_encoder = resources["label_encoder"]
df = resources["df"]
dev_counts = resources["dev_counts"]
publisher_counts = resources["publisher_counts"]
dev_threshold = resources["dev_threshold"]
publisher_threshold = resources["publisher_threshold"]
reference_text_matrix = resources["reference_text_matrix"]
explanation_kb = resources["explanation_kb"]

numeric_cols = feature_config["numeric_cols"]
boolean_cols = feature_config["boolean_cols"]
genre_cols = feature_config["genre_cols"]
tag_cols = feature_config["tag_cols"]
cat_cols = feature_config["cat_cols"]
tfidf_col = feature_config["tfidf_col"]

genre_lookup = {col.replace("genre_", "").lower(): col for col in genre_cols}
tag_lookup = {col.replace("tag_", "").lower(): col for col in tag_cols}
cat_lookup = {
    col.replace("cat_", "").replace("_", " ").lower(): col
    for col in cat_cols
}
cat_lookup["single-player"] = "cat_single_player"
cat_lookup["multi-player"] = "cat_multi_player"
cat_lookup["co-op"] = "cat_co_op"
cat_lookup["online co-op"] = "cat_online_co_op"
cat_lookup["local co-op"] = "cat_local_co_op"
cat_lookup["shared/split screen"] = "cat_shared_split_screen"

genre_options = [col.replace("genre_", "") for col in genre_cols]
tag_options = [col.replace("tag_", "") for col in tag_cols]
category_options = [format_category_label(col) for col in cat_cols]
category_to_col = dict(zip(category_options, cat_cols))


def get_dev_score(dev_list):
    if not dev_list:
        return 0
    return max([dev_counts.get(dev, 0) for dev in dev_list])


def get_publisher_score(publisher_list):
    if not publisher_list:
        return 0
    return max([dev_counts.get(pub, 0) for pub in publisher_list])


def prepare_input(game_data):
    prepared = {}

    prepared["short_description"] = game_data["short_description"]
    prepared["short_description_clean"] = clean_description(game_data["short_description"])

    prepared["required_age"] = game_data.get("required_age", 0)
    prepared["release_year"] = game_data.get("release_year", 2025)

    prepared["windows"] = int(game_data.get("windows", 1))
    prepared["mac"] = int(game_data.get("mac", 0))
    prepared["linux"] = int(game_data.get("linux", 0))

    achievements = game_data.get("achievements", 0)
    num_supported_languages = game_data.get("num_supported_languages", 0)
    num_audio_languages = game_data.get("num_audio_languages", 0)

    developers = to_list(game_data.get("developers", []))
    publishers = to_list(game_data.get("publishers", []))

    developer_presence = get_dev_score(developers)
    publisher_presence = get_publisher_score(publishers)

    prepared["log_achievements"] = np.log1p(achievements)
    prepared["log_num_supported_languages"] = np.log1p(num_supported_languages)
    prepared["log_num_audio_languages"] = np.log1p(num_audio_languages)
    prepared["log_developer_presence"] = np.log1p(developer_presence)
    prepared["log_publisher_presence"] = np.log1p(publisher_presence)

    prepared["top_developer"] = int(developer_presence >= dev_threshold)
    prepared["top_publisher"] = int(publisher_presence >= publisher_threshold)

    for col in genre_cols + tag_cols + cat_cols:
        prepared[col] = 0

    for genre in to_list(game_data.get("genres", [])):
        key = str(genre).strip().lower()
        if key in genre_lookup:
            prepared[genre_lookup[key]] = 1

    for tag in to_list(game_data.get("tags", [])):
        key = str(tag).strip().lower()
        if key in tag_lookup:
            prepared[tag_lookup[key]] = 1

    for category in to_list(game_data.get("categories", [])):
        key = str(category).strip().lower()
        if key in cat_lookup:
            prepared[cat_lookup[key]] = 1

    return prepared


def preprocess_input(game_data):
    input_df = pd.DataFrame([game_data])
    x_num = scaler.transform(input_df[numeric_cols])
    x_bool = input_df[boolean_cols].astype(int).values
    x_cat = input_df[genre_cols + tag_cols + cat_cols].astype(int).values
    x_text = tfidf.transform(input_df[tfidf_col])
    return hstack([x_num, x_bool, x_cat, x_text])


def predict_game(game_data):
    prepared_game = prepare_input(game_data)
    x = preprocess_input(prepared_game)

    free_prob = float(stage1_model.predict_proba(x)[:, 1][0])

    if free_prob >= threshold:
        return {
            "type": "Free",
            "free_probability": free_prob,
            "paid_probability": 1 - free_prob,
        }, prepared_game, x

    stage2_probs = stage2_model.predict_proba(x)[0]
    pred_idx = int(stage2_probs.argmax())
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return {
        "type": "Paid",
        "free_probability": free_prob,
        "paid_probability": 1 - free_prob,
        "price_tier": pred_label,
        "probs": stage2_probs,
        "labels": list(label_encoder.classes_),
    }, prepared_game, x


def extract_active_features(game_data):
    active_features = []
    text = game_data["short_description_clean"].lower()

    for col in genre_cols:
        if game_data.get(col, 0) == 1:
            active_features.append(col.lower())

    for col in tag_cols:
        if game_data.get(col, 0) == 1:
            active_features.append(col.lower())

    for col in cat_cols:
        if game_data.get(col, 0) == 1:
            active_features.append(col.lower())

    if game_data.get("top_developer", 0) == 1:
        active_features.append("top_developer")

    if game_data.get("top_publisher", 0) == 1:
        active_features.append("top_publisher")

    if game_data["log_achievements"] > np.log1p(20):
        active_features.append("log_achievements")

    if game_data["log_num_supported_languages"] > np.log1p(8):
        active_features.append("log_num_supported_languages")

    if game_data["log_num_audio_languages"] > np.log1p(4):
        active_features.append("log_num_audio_languages")

    for key in [
        "open world",
        "multiplayer",
        "story",
        "combat",
        "battle royale",
        "live service",
        "simulation",
        "crafting",
    ]:
        if key in text:
            active_features.append(key)

    return list(dict.fromkeys(active_features))


def retrieve_explanations(active_features):
    explanations = []
    for feature in active_features:
        if feature in explanation_kb:
            explanations.append(explanation_kb[feature])
    return explanations


def retrieve_similar_games(game_data, result, top_k=5):
    query_vector = tfidf.transform([game_data["short_description_clean"]])
    similarities = cosine_similarity(query_vector, reference_text_matrix).flatten()

    temp_df = df[["short_description", "price", "price_category", "is_free"]].copy()
    temp_df["similarity"] = similarities

    result_type = str(result.get("type", "")).strip().lower()

    if result_type == "free":
        temp_df = temp_df[temp_df["is_free"] == 1].copy()
    elif result_type == "paid" and "price_tier" in result:
        paid_same_tier = temp_df[
            (temp_df["is_free"] == 0)
            & (temp_df["price_category"] == result["price_tier"])
        ].copy()

        if len(paid_same_tier) >= top_k:
            temp_df = paid_same_tier
        else:
            temp_df = temp_df[temp_df["is_free"] == 0].copy()

    temp_df = temp_df.sort_values("similarity", ascending=False).head(top_k)
    return temp_df.reset_index(drop=True)


def generate_explanation(result, explanations, similar_games):
    text = ""

    if result["type"] == "Free":
        text += "PREDICTION: FREE GAME\n\n"
        text += f"Stage 01 Free Probability: {result['free_probability']:.3f}\n"
        text += f"Stage 01 Paid Probability: {result['paid_probability']:.3f}\n"
        text += f"Decision Threshold: {threshold:.3f}\n"
    else:
        tier_map = {
            "budget": "0.01 - 0.99 USD",
            "low": "1.00 - 4.99 USD",
            "mid": "5.00 - 9.99 USD",
            "premium": "> 9.99 USD",
        }

        text += f"PREDICTION: {result['price_tier'].upper()} PRICE GAME\n\n"
        text += f"Recommended Price Band: {tier_map[result['price_tier']]}\n"
        text += f"Stage 01 Free Probability: {result['free_probability']:.3f}\n"
        text += f"Stage 01 Paid Probability: {result['paid_probability']:.3f}\n\n"

        text += "Stage 02 Class Probabilities:\n"
        for label, prob in zip(result["labels"], result["probs"]):
            text += f"- {label}: {prob:.3f}\n"

    if explanations:
        text += "\nRetrieved Explanation Signals:\n"
        for exp in explanations:
            text += f"- {exp}\n"

    if len(similar_games) > 0:
        text += "\nRetrieved Similar Games:\n"
        for _, row in similar_games.head(3).iterrows():
            text += (
                f"- {row['price_category']} | ${row['price']:.2f} | "
                f"similarity={row['similarity']:.3f} | "
                f"{str(row['short_description'])[:110]}\n"
            )

        if result["type"] == "Paid":
            text += f"\nAverage retrieved price: ${similar_games['price'].mean():.2f}"

    return text


def run_pipeline(game_data):
    result, prepared_game, x = predict_game(game_data)
    active_features = extract_active_features(prepared_game)
    explanations = retrieve_explanations(active_features)
    similar_games = retrieve_similar_games(prepared_game, result)
    final_text = generate_explanation(result, explanations, similar_games)
    return final_text, result, prepared_game, similar_games


st.markdown(
    """
    <div class="hero">
        <h1>Steam Price Intelligence System</h1>
        <p>Notebook-based pricing inference and RAG explanation layer, wrapped into a Streamlit app for user input, prediction, and explanation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Input Source")
    selected_mode = st.selectbox("Choose input mode", list(EXAMPLES.keys()))
    st.caption("The app mirrors the logic from `05_Rag_Explanation.ipynb`.")

    if selected_mode != "Custom input":
        st.info("Using a preset example from the notebook.")


if selected_mode == "Custom input":
    st.subheader("Game Input")
    col1, col2 = st.columns([1.35, 1.0])

    with col1:
        short_description = st.text_area(
            "Short Description",
            height=180,
            placeholder="Describe the game in a Steam-style short description...",
        )

    with col2:
        age_col, year_col = st.columns(2)
        with age_col:
            required_age = st.number_input("Required Age", min_value=0, max_value=21, value=12)
        with year_col:
            release_year = st.number_input("Release Year", min_value=2010, max_value=2035, value=2026)

        ach_col, lang_col, audio_col = st.columns(3)
        with ach_col:
            achievements = st.number_input("Achievements", min_value=0, max_value=500, value=10)
        with lang_col:
            num_supported_languages = st.number_input("Supported Languages", min_value=0, max_value=50, value=8)
        with audio_col:
            num_audio_languages = st.number_input("Audio Languages", min_value=0, max_value=30, value=2)

        developers = st.text_input("Developers", placeholder="Comma-separated, e.g. FromSoftware")
        publishers = st.text_input("Publishers", placeholder="Comma-separated, e.g. Bandai Namco")

    st.subheader("Platforms")
    platform_col1, platform_col2, platform_col3 = st.columns(3)
    with platform_col1:
        windows = st.checkbox("Windows", value=True)
    with platform_col2:
        mac = st.checkbox("Mac", value=False)
    with platform_col3:
        linux = st.checkbox("Linux", value=False)

    st.subheader("Genres, Tags, and Categories")
    genre_values = st.multiselect("Genres", genre_options)
    tag_values = st.multiselect("Tags", tag_options)
    category_values = st.multiselect("Categories", category_options)

    game_input = {
        "short_description": short_description,
        "required_age": int(required_age),
        "release_year": int(release_year),
        "achievements": int(achievements),
        "num_supported_languages": int(num_supported_languages),
        "num_audio_languages": int(num_audio_languages),
        "developers": to_list(developers),
        "publishers": to_list(publishers),
        "windows": int(windows),
        "mac": int(mac),
        "linux": int(linux),
        "genres": genre_values,
        "tags": tag_values,
        "categories": category_values,
    }
else:
    game_input = EXAMPLES[selected_mode]
    st.subheader("Selected Example")
    st.json(game_input)


predict_clicked = st.button("Predict and Explain", type="primary", use_container_width=True)


if predict_clicked:
    if not game_input["short_description"].strip():
        st.warning("Please add a short description before running the pipeline.")
        st.stop()

    with st.spinner("Running Steam pricing pipeline..."):
        explanation_text, result, prepared_game, similar_games = run_pipeline(game_input)

    st.subheader("Prediction Output")
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    final_label = result["type"] if result["type"] == "Free" else f"Paid - {result['price_tier'].title()}"
    metric_col1.markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_col1.metric("Prediction", final_label)
    metric_col1.markdown("</div>", unsafe_allow_html=True)

    metric_col2.markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_col2.metric("Free Probability", f"{result['free_probability']:.3f}")
    metric_col2.markdown("</div>", unsafe_allow_html=True)

    metric_col3.markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_col3.metric("Paid Probability", f"{result['paid_probability']:.3f}")
    metric_col3.markdown("</div>", unsafe_allow_html=True)

    if result["type"] == "Paid":
        st.subheader("Stage 02 Price Tier Probabilities")
        prob_df = pd.DataFrame(
            {
                "price_tier": result["labels"],
                "probability": result["probs"],
            }
        ).sort_values("probability", ascending=False)
        st.bar_chart(prob_df.set_index("price_tier"))
        st.dataframe(
            prob_df.style.format({"probability": "{:.3f}"}),
            use_container_width=True,
        )

    st.subheader("Explanation")
    st.code(explanation_text, language="text")

    st.subheader("Similar Historical Games")
    if len(similar_games) > 0:
        display_df = similar_games.copy()
        display_df["price"] = display_df["price"].map(lambda x: f"${x:,.2f}")
        display_df["similarity"] = display_df["similarity"].map(lambda x: f"{x:.3f}")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No similar games were retrieved.")

    with st.expander("Prepared Model Features"):
        prepared_df = pd.DataFrame([prepared_game]).T.reset_index()
        prepared_df.columns = ["feature", "value"]
        st.dataframe(prepared_df, use_container_width=True, hide_index=True)
