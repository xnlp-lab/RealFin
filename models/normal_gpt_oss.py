# -*- coding: utf-8 -*-
"""
GPT-OSS MCQ Evaluation - Normal version
Supports both GPT-OSS-20B and GPT-OSS-120B (set MODEL_SIZE below).
"""

import os
import re
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_SIZE = "20b"   # change to "120b" for the larger model

_MODEL_IDS = {
    "20b":  "openai/gpt-oss-20b",
    "120b": "openai/gpt-oss-120b",
}
assert MODEL_SIZE in _MODEL_IDS, f"MODEL_SIZE must be one of {list(_MODEL_IDS)}"
MODEL_ID = _MODEL_IDS[MODEL_SIZE]
MODEL_NAME = f"gpt_oss_{MODEL_SIZE}"

INPUT_PATHS = [
    os.path.abspath("./missing_questions_Chinese.csv"),
    os.path.abspath("./missing_questions_English.csv"),
    os.path.abspath("./original_questions_Chinese.csv"),
    os.path.abspath("./original_questions_English.csv"),
]

OUTPUT_DIR = os.path.abspath(f"./outputs_{MODEL_NAME}")
OUTPUT_PREFIX = "questions_with_pred"

COLUMN_MAPPINGS = {
    "Question": ["Question", "question"],
    "Option A": ["Option A", "option_a", "A"],
    "Option B": ["Option B", "option_b", "B"],
    "Option C": ["Option C", "option_c", "C"],
    "Option D": ["Option D", "option_d", "D"],
    "Correct Answer": ["Correct Answer", "correct_answer", "Answer", "answer"],
}

# Generation hyperparameters (frozen for reproducibility)
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.05
DO_SAMPLE = False



SYSTEM_PROMPT_CN = "您是一位金融问题解决专家。您将收到一道单选或多选题。请返回一个严格的 JSON 数据，包含三个键：\"reason\" 和 \"answer\" 和 \"confidence\"。"

USER_PROMPT_TEMPLATE_CN = """问题：
{question}

请严格按照以下格式输出 JSON 数据。请勿在 JSON 对象之外包含任何文本。
{{
 "reason": "简要解释这些选项正确的原因。",
 "answer": "仅输出大写字母 (A-Z)。对于多选题，请返回按字母顺序排列并连接起来的字母集合（例如，ACD）。对于单选题，请返回字母，例如 A 或 B 或 C 或 D。",
 "confidence": "你对本题你所回答的答案的信心程度是多少，在0-100里选一个。"
}}"""

SYSTEM_PROMPT_EN = "You are an expert in financial problem solving. You will be given a single- or multi-choice question. Please return STRICT JSON with three keys: \"reason\", \"answer\" and \"confidence\"."

USER_PROMPT_TEMPLATE_EN = """Question:
{question}

Output JSON EXACTLY in the following format. Do NOT include any text outside the JSON object.
{{
 "reason": "a concise explanation of why these options are correct.",
 "answer": "output uppercase letters only (A-Z). For multi-choice, return the set of letters sorted alphabetically and concatenated (e.g., ACD). For single-choice, return letter like A or B or C or D.",
 "confidence": "How confident are you in your answer to this question? Choose a score between 0 and 100."
}}"""


def read_table(path: str) -> pd.DataFrame:
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
        if header[:4] == b'PK\x03\x04':
            print(f"  [INFO] Detected Excel format for: {os.path.basename(path)}")
            return pd.read_excel(path)
    except Exception:
        pass
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    for enc in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="ignore")


def find_column(df: pd.DataFrame, standard_name: str):
    if standard_name in COLUMN_MAPPINGS:
        for v in COLUMN_MAPPINGS[standard_name]:
            if v in df.columns:
                return v
    return standard_name if standard_name in df.columns else None


def normalize_columns(df: pd.DataFrame):
    col_map = {}
    missing = []
    for std_name in COLUMN_MAPPINGS.keys():
        actual = find_column(df, std_name)
        if actual:
            col_map[std_name] = actual
        else:
            missing.append(std_name)
    return col_map, missing


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_answerset(x: str) -> str:
    """
    Restricted to [A-D] because the benchmark only has 4 options.
    Avoids false-positive matches on incidental uppercase letters
    (e.g., 'ROE', 'GDP', 'CFA') in natural-language reasoning text.
    """
    s = safe_str(x).upper()
    letters = re.findall(r"[A-D]", s)
    if not letters:
        return ""
    return "".join(sorted(set(letters)))


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Pred_Choice", "Why_Choice", "Is_Correct", "Confidence", "Raw_Model_Output"]
    for c in cols:
        if c not in df.columns:
            if c == "Is_Correct":
                df[c] = False
            elif c == "Confidence":
                df[c] = 0.0
            else:
                df[c] = ""
    for c in ["Pred_Choice", "Why_Choice", "Raw_Model_Output"]:
        df[c] = df[c].astype("object")
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce").fillna(0.0)
    df["Is_Correct"] = df["Is_Correct"].astype(bool)
    return df


def save_df_atomic(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def load_or_init_working_df(input_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    if os.path.exists(output_path):
        try:
            out_df = pd.read_csv(output_path, encoding="utf-8-sig")
        except Exception:
            out_df = pd.read_csv(output_path, encoding="utf-8", errors="ignore")
        return ensure_output_columns(out_df)
    return ensure_output_columns(input_df.copy())


def row_already_done(row) -> bool:
    pred = normalize_answerset(row.get("Pred_Choice", ""))
    why = safe_str(row.get("Why_Choice", ""))
    if why.startswith("ERROR"):
        return False
    conf = row.get("Confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    return len(pred) > 0 and len(why) > 0 and conf > 0


def is_chinese_dataset(path: str) -> bool:
    return "chinese" in os.path.basename(path).lower()



def build_prompt_chattemplated(tokenizer, question, a, b, c, d, is_chinese: bool):
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template")
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError("Tokenizer chat_template is empty/missing")

    if is_chinese:
        question_block = f"{question}\n\nA：{a}\nB：{b}\nC：{c}\nD：{d}"
        system_msg = SYSTEM_PROMPT_CN
        user_msg = USER_PROMPT_TEMPLATE_CN.format(question=question_block)
    else:
        question_block = f"{question}\n\nA: {a}\nB: {b}\nC: {c}\nD: {d}"
        system_msg = SYSTEM_PROMPT_EN
        user_msg = USER_PROMPT_TEMPLATE_EN.format(question=question_block)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt



def _regex_field(raw: str, key: str) -> str:
    m = re.search(
        rf'["\']{re.escape(key)}["\']\s*:\s*("([^"]*)"|\'([^\']*)\'|([0-9]+(\.[0-9]+)?))',
        raw, re.IGNORECASE,
    )
    if not m:
        return ""
    for g in [2, 3, 4]:
        if m.group(g) is not None:
            return m.group(g)
    return ""


def _to_confidence(x) -> float:
    s = safe_str(x)
    if not s:
        return 0.0
    m = re.search(r"([0-9]+(\.[0-9]+)?)", s)
    if not m:
        return 0.0
    try:
        return max(0.0, min(100.0, float(m.group(1))))
    except Exception:
        return 0.0


def _extract_fields(obj: dict):
    ans = normalize_answerset(obj.get("answer", ""))
    rea = safe_str(obj.get("reason", "")) or "(reason not parsed)"
    conf = _to_confidence(obj.get("confidence", ""))
    ok = len(ans) > 0
    return ans, rea, conf, ok


def parse_json_from_text(text: str):
    """
    Harmony-aware JSON parser. GPT-OSS emits 'analysis' (CoT) and 'final'
    channels; the JSON answer sits in the final channel. After
    skip_special_tokens=True, channels appear as plain text prefixes.

    (1) direct json.loads (rare - whole response is strict JSON)
    (2) last-{...} substring (covers both 'final' JSON and 'analysis' with trailing JSON)
    (3) regex fallback on the three keys
    """
    raw = safe_str(text)
    if not raw:
        return "", "(empty output)", 0.0, False

    try:
        return _extract_fields(json.loads(raw))
    except Exception:
        pass

    l = raw.find("{")
    r = raw.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return _extract_fields(json.loads(raw[l:r + 1]))
        except Exception:
            pass

    ans = normalize_answerset(_regex_field(raw, "answer"))
    rea = safe_str(_regex_field(raw, "reason")) or "(reason not parsed)"
    conf = _to_confidence(_regex_field(raw, "confidence"))
    ok = len(ans) > 0
    return ans, rea, conf, ok



def onepass_generate(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
            return_dict_in_generate=True,
            output_scores=False,
        )
    seq = out.sequences[0]
    gen_tokens = seq[input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()



def process_dataset(input_path: str, output_path: str, model, tokenizer, device):
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"Output:     {os.path.basename(output_path)}")
    print(f"{'='*60}")

    input_df = read_table(input_path)
    print(f"  Loaded {len(input_df)} rows")

    col_map, missing = normalize_columns(input_df)
    if missing:
        print(f"  [ERROR] Missing columns: {missing}")
        print(f"  Available columns: {list(input_df.columns)}")
        return
    print(f"  Column mapping: {col_map}")

    df = load_or_init_working_df(input_df, output_path)
    if len(df) != len(input_df):
        print(f"  [ERROR] Row count mismatch. Delete {output_path} and retry.")
        return

    is_chinese = is_chinese_dataset(input_path)
    lang = "Chinese" if is_chinese else "English"
    print(f"  Language: {lang}")

    remaining = sum(1 for idx in range(len(df)) if not row_already_done(df.iloc[idx]))
    print(f"  Remaining: {remaining}/{len(df)}")
    if remaining == 0:
        print("  All rows already processed. Skipping.")
        return

    col_q = col_map["Question"]
    col_a = col_map["Option A"]
    col_b = col_map["Option B"]
    col_c = col_map["Option C"]
    col_d = col_map["Option D"]
    col_gt = col_map["Correct Answer"]

    for idx in tqdm(range(len(df)), desc=f"  GPT-OSS-{MODEL_SIZE} ({lang})"):
        row = df.iloc[idx]
        if row_already_done(row):
            continue

        q = safe_str(row[col_q])
        a = safe_str(row[col_a])
        b = safe_str(row[col_b])
        c = safe_str(row[col_c])
        d = safe_str(row[col_d])
        gt = normalize_answerset(row[col_gt])

        try:
            prompt = build_prompt_chattemplated(tokenizer, q, a, b, c, d, is_chinese)
            after_text = onepass_generate(model, tokenizer, prompt, device)

            pred, reason, conf, ok = parse_json_from_text(after_text)
            if not ok:
                pred = ""
                conf = 0.0
                reason = "(Model output not parseable; see Raw_Model_Output)"

            correct = (pred == gt) if gt else False

            df.at[idx, "Pred_Choice"] = pred
            df.at[idx, "Why_Choice"] = reason
            df.at[idx, "Is_Correct"] = bool(correct)
            df.at[idx, "Confidence"] = float(conf)
            df.at[idx, "Raw_Model_Output"] = after_text
        except Exception as e:
            err = str(e)
            print(f"  [ERROR] row {idx}: {err[:150]}")
            df.at[idx, "Pred_Choice"] = ""
            df.at[idx, "Why_Choice"] = f"ERROR: {err[:300]}"
            df.at[idx, "Is_Correct"] = False
            df.at[idx, "Confidence"] = 0.0
            df.at[idx, "Raw_Model_Output"] = ""
            if "CUDA" in err or "out of memory" in err.lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_df_atomic(df, output_path)

    total = len(df)
    correct_count = int(df["Is_Correct"].sum())
    accuracy = correct_count / total * 100 if total > 0 else 0
    print(f"\n  Completed: {os.path.basename(input_path)}")
    print(f"  Accuracy:  {correct_count}/{total} = {accuracy:.2f}%")



def main():
    print("=" * 60)
    print(f"GPT-OSS-{MODEL_SIZE} MCQ Evaluation (Normal)")
    print(f"Model: {MODEL_ID}")
    print(f"Decoding: do_sample={DO_SAMPLE}, rep_penalty={REPETITION_PENALTY}, "
          f"max_new_tokens={MAX_NEW_TOKENS}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading tokenizer & model from: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "Tokenizer chat_template is empty/missing. "
            "GPT-OSS ships with a built-in Harmony chat template."
        )
    print("  Chat template: OK (Harmony, model's built-in)")

    # torch_dtype="auto" is required for MXFP4 MoE weights; transformers>=4.55
    # will honour the native quantisation config shipped with the checkpoint.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    print("  Model loaded successfully")

    device = model.device if hasattr(model, "device") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  Device: {device}")

    for input_path in INPUT_PATHS:
        if not os.path.exists(input_path):
            print(f"\n[WARNING] File not found, skipping: {input_path}")
            continue
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(
            OUTPUT_DIR, f"{OUTPUT_PREFIX}_{basename}_{MODEL_NAME}.csv"
        )
        process_dataset(input_path, output_path, model, tokenizer, device)

    print("\n" + "=" * 60)
    print("All datasets processed.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
