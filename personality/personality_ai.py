import os, json, random, re
import numpy as np, pandas as pd
import faiss, emoji
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from collections import Counter
from openai import OpenAI


# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "father_son_chat_2021.json")

# Choose any OpenRouter-supported model:
# Examples:
# "openai/gpt-4.1-mini"
# "anthropic/claude-3.5-sonnet"
# "mistralai/mixtral-8x7b-instruct"
OPENROUTER_MODEL = "openai/gpt-4.1-mini"


# ---------------- OpenRouter Client ----------------
def init_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY as environment variable")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    return client

client = init_openrouter()


# ---------------- LOAD DATASET ----------------
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

msgs = data.get("messages", data) if isinstance(data, dict) else data
df = pd.DataFrame(msgs)

dad_df = df[df["sender"].astype(str).str.lower() == "dad"].reset_index(drop=True)


# ---------------- CLEANING ----------------
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.strip()
    t = emoji.replace_emoji(t, replace="")
    t = re.sub(r"[\u200b\u200e]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

text_col = None
for cand in ["text_original", "text", "message", "content"]:
    if cand in dad_df.columns:
        text_col = cand
        break

dad_df["clean"] = dad_df[text_col].apply(clean_text)
dad_msgs = dad_df["clean"].tolist()


# ---------------- EMBEDDINGS + FAISS ----------------
embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
dad_df["embedding"] = dad_df["clean"].apply(lambda t: embedder.encode(str(t)).astype("float32"))
dad_emb = np.vstack(dad_df["embedding"].values)

index = faiss.IndexFlatL2(dad_emb.shape[1])
index.add(dad_emb)


# ---------------- BM25 ----------------
corpus_tokens = [msg.lower().split() for msg in dad_msgs]
bm25 = BM25Okapi(corpus_tokens)


# ---------------- STYLE ----------------
def analyze_style(texts):
    lengths = [len(t.split()) for t in texts if t.strip()]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    starters = Counter([t.split()[0].lower() for t in texts if t.strip()])
    top_starters = [s for s, _ in starters.most_common(10)]
    return {"avg_len": avg_len, "top_starters": top_starters}

STYLE_SUMMARY = analyze_style(dad_msgs)


# ---------------- RETRIEVAL ----------------
def retrieve_hybrid(query, k_sem=6, k_lex=6):
    q_emb = embedder.encode(query).astype("float32")
    _, I = index.search(np.array([q_emb]), k_sem)
    sem_ids = set(I[0].tolist())

    bm_scores = bm25.get_scores(query.lower().split())
    lex_ids = set(np.argsort(bm_scores)[-k_lex:])

    final_ids = list(sem_ids.union(lex_ids))
    final_ids = [i for i in final_ids if 0 <= i < len(dad_df)]

    if not final_ids:
        return dad_df.sample(min(3, len(dad_df)))
    return dad_df.iloc[final_ids].sample(min(5, len(final_ids)))


# ---------------- PROMPT ----------------
def persona_prompt_text():
    return (
        "You are replying AS the deceased person's voice based on the examples provided. "
        "Match tone, length and style. "
        "Avoid dependency phrases, medical or legal advice. "
        "Keep replies short (1-4 lines)."
    )

def messages_to_prompt(messages):
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("DAD REPLY:")
    return "\n\n".join(parts)

def build_prompt(user_msg):
    retrieved = retrieve_hybrid(user_msg)
    messages = [{"role": "system", "content": persona_prompt_text()}]
    for _, r in retrieved.head(3).iterrows():
        messages.append({"role": "assistant", "content": r["clean"]})
    messages.append({"role": "user", "content": user_msg})
    return messages


# ---------------- POST PROCESS ----------------
THERAPY_BAN = ["time heals", "therapy", "medical advice"]
DEPENDENCY_BAN = ["always here for you"]
THIRDPERSON_BAN = ["your dad", "he was"]

def clean_reply(txt):
    t = str(txt)
    for ban in THERAPY_BAN + DEPENDENCY_BAN + THIRDPERSON_BAN:
        t = re.sub(re.escape(ban), "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


# ---------------- LLM CALL ----------------
def openrouter_generate(prompt_text):
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": persona_prompt_text()},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.6,
        max_tokens=150,
    )
    return resp.choices[0].message.content


# ---------------- MAIN FUNCTION ----------------
def generate_personality_reply(user_text):
    prompt_msgs = build_prompt(user_text)
    prompt_text = messages_to_prompt(prompt_msgs)

    raw = openrouter_generate(prompt_text)
    final = clean_reply(raw)

    return final


# ---------------- Standalone Test ----------------
if __name__ == "__main__":
    test = "I feel like nothing is improving, I miss him a lot."
    print(generate_personality_reply(test))
