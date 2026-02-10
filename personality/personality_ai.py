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

# Best model for personality cloning - Claude 3.5 Sonnet excels at:
# - Capturing nuanced speaking patterns
# - Maintaining consistent persona
# - Understanding emotional context
# Alternative: "openai/gpt-4o" (also excellent for personality)
OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"


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

# Extract persona metadata if available
PERSONA_META = data.get("meta", {}) if isinstance(data, dict) else {}
PERSONA_DESCRIPTION = PERSONA_META.get("persona", "Calm, patient, affectionate, motivational")

msgs = data.get("messages", data) if isinstance(data, dict) else data
df = pd.DataFrame(msgs)

dad_df = df[df["sender"].astype(str).str.lower() == "dad"].reset_index(drop=True)


# ---------------- CLEANING ----------------
def clean_text(t, preserve_emoji=False):
    """Clean text while optionally preserving emojis for personality."""
    if not isinstance(t, str):
        return ""
    t = t.strip()
    if not preserve_emoji:
        t = emoji.replace_emoji(t, replace="")
    t = re.sub(r"[\u200b\u200e]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

text_col = None
for cand in ["text_original", "text", "message", "content"]:
    if cand in dad_df.columns:
        text_col = cand
        break

# Keep original with emojis for personality, clean version for search
dad_df["clean"] = dad_df[text_col].apply(lambda t: clean_text(t, preserve_emoji=False))
dad_df["original"] = dad_df[text_col].apply(lambda t: clean_text(t, preserve_emoji=True))
dad_msgs = dad_df["clean"].tolist()
dad_originals = dad_df["original"].tolist()


# ---------------- ADDRESSING PATTERNS ----------------
def extract_addressing_patterns(texts):
    """Extract how the father addresses his son (kanna, ra, beta, etc.)."""
    patterns = []
    # Common Telugu/Hindi affectionate terms
    address_regex = r'\b(kanna|ra|beta|champ|nanna|babu)\b'
    
    for text in texts:
        if isinstance(text, str):
            matches = re.findall(address_regex, text.lower())
            patterns.extend(matches)
    
    return Counter(patterns).most_common(10)

ADDRESSING_PATTERNS = extract_addressing_patterns(dad_originals)
TOP_ADDRESS_TERMS = [term for term, _ in ADDRESSING_PATTERNS[:5]] if ADDRESSING_PATTERNS else ["beta"]


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
    """Analyze speaking style: length, starters, common phrases."""
    lengths = [len(t.split()) for t in texts if t.strip()]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    starters = Counter([t.split()[0].lower() for t in texts if t.strip()])
    top_starters = [s for s, _ in starters.most_common(10)]
    
    # Extract common phrases (2-3 word patterns)
    all_words = " ".join(texts).lower()
    common_phrases = []
    phrase_patterns = [
        r'\b(don\'?t worry)\b', r'\b(proud of you)\b', r'\b(step by step)\b',
        r'\b(small wins)\b', r'\b(one at a time)\b', r'\b(I\'?m here)\b',
        r'\b(keep going)\b', r'\b(that\'?s okay)\b', r'\b(chala bagundi)\b'
    ]
    for pattern in phrase_patterns:
        if re.search(pattern, all_words, re.I):
            match = re.search(pattern, all_words, re.I)
            if match:
                common_phrases.append(match.group(1))
    
    return {
        "avg_len": avg_len,
        "top_starters": top_starters,
        "common_phrases": common_phrases
    }

STYLE_SUMMARY = analyze_style(dad_originals)


# ---------------- RETRIEVAL ----------------
def retrieve_hybrid(query, k_sem=6, k_lex=6):
    """Hybrid retrieval using semantic + lexical search."""
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
    """Build detailed persona prompt with addressing patterns and style."""
    address_terms = ", ".join(TOP_ADDRESS_TERMS) if TOP_ADDRESS_TERMS else "beta, kanna"
    avg_words = int(STYLE_SUMMARY.get("avg_len", 15))
    
    return f"""You are replying AS a loving father to his son, based on the real chat examples provided.

PERSONALITY: {PERSONA_DESCRIPTION}

CRITICAL - ADDRESSING PATTERNS:
- Use terms of endearment like: {address_terms}
- Mix Telugu (romanized) with English naturally, as shown in examples
- Examples: "kanna" (dear child), "ra" (affectionate suffix), "beta" (son)

SPEAKING STYLE:
- Keep replies SHORT: {avg_words-5} to {avg_words+10} words typical
- Be warm, supportive, and practical
- Offer gentle wisdom without being preachy
- Use occasional emojis sparingly (😊, 😄) like in examples

STRICT RULES:
- NEVER use third person ("your dad", "he would say")
- NEVER give medical/legal advice
- NEVER use dependency phrases like "always here for you"
- Speak DIRECTLY as the father, in first person
- Match the exact tone and vocabulary from the examples below"""

def messages_to_prompt(messages):
    """Convert messages to a formatted prompt string."""
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("DAD REPLY:")
    return "\n\n".join(parts)

def build_prompt(user_msg):
    """Build prompt with retrieved examples showing the father's authentic voice."""
    retrieved = retrieve_hybrid(user_msg)
    messages = [{"role": "system", "content": persona_prompt_text()}]
    
    # Add example messages from the actual chat history (use original with emojis)
    examples_text = "\n\nEXAMPLES OF HOW THE FATHER SPEAKS:"
    for _, r in retrieved.head(5).iterrows():
        # Use original text to preserve personality markers like emojis
        original_text = r.get("original", r["clean"])
        examples_text += f"\n- \"{original_text}\""
    
    messages.append({"role": "system", "content": examples_text})
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
def openrouter_generate(messages):
    """Generate response using OpenRouter API with structured messages."""
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=0.7,  # Slightly higher for more natural personality
        max_tokens=150,
    )
    return resp.choices[0].message.content


# ---------------- MAIN FUNCTION ----------------
def generate_personality_reply(user_text):
    """Generate a reply that matches the father's personality and addressing patterns."""
    prompt_msgs = build_prompt(user_text)
    
    raw = openrouter_generate(prompt_msgs)
    final = clean_reply(raw)

    return final


# ---------------- Standalone Test ----------------
if __name__ == "__main__":
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"Persona: {PERSONA_DESCRIPTION}")
    print(f"Addressing patterns: {TOP_ADDRESS_TERMS}")
    print(f"Average message length: {STYLE_SUMMARY.get('avg_len', 0):.1f} words")
    print("-" * 50)
    
    test_messages = [
        "I feel like nothing is improving, I miss him a lot.",
        "Dad I'm feeling low today",
        "I failed my exam",
    ]
    
    for test in test_messages:
        print(f"\nUser: {test}")
        print(f"Dad: {generate_personality_reply(test)}")
