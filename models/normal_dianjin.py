

import os
import re
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_DIR = os.path.abspath("./model/DianJin-R1-32B")

INPUT_PATHS = [
    os.path.abspath("./missing_questions_Chinese.csv"),
    os.path.abspath("./missing_questions_English.csv"),
    os.path.abspath("./original_questions_Chinese.csv"),
    os.path.abspath("./original_questions_English.csv")
]

OUTPUT_DIR = os.path.abspath("./outputs_dianjin_r1_32b")
OUTPUT_PREFIX = "questions_with_pred"

COLUMN_MAPPINGS = {
    "Question": ["Question", "question"],
    "Option A": ["Option A", "option_a", "A"],
    "Option B": ["Option B", "option_b", "B"],
    "Option C": ["Option C", "option_c", "C"],
    "Option D": ["Option D", "option_d", "D"],
    "Correct Answer": ["Correct Answer", "correct_answer", "Answer", "answer"],
}

MAX_NEW_TOKENS = 1024



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


def find_column(df: pd.DataFrame, standard_name: str) -> str:
    if standard_name in COLUMN_MAPPINGS:
        for v in COLUMN_MAPPINGS[standard_name]:
            if v in df.columns:
                return v
    return standard_name if standard_name in df.columns else None


def normalize_columns(df: pd.DataFrame) -> tuple:
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
    s = safe_str(x).upper()
    letters = re.findall(r"[A-D]", s)
    if not letters:
        return ""
    return "".join(sorted(set(letters)))


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Pred_Choice", "Why_Choice", "Is_Correct", "Confidence", "Raw_Model_Output"]
    for c in cols:
        if c not in df.columns:
            if c in ["Pred_Choice", "Why_Choice", "Raw_Model_Output"]:
                df[c] = ""
            elif c == "Is_Correct":
                df[c] = False
            else:
                df[c] = 0.0
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
        out_df = ensure_output_columns(out_df)
        return out_df
    df = input_df.copy()
    df = ensure_output_columns(df)
    return df


def row_already_done(row) -> bool:
    pred = normalize_answerset(row.get("Pred_Choice", ""))
    why = safe_str(row.get("Why_Choice", ""))
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
        add_generation_prompt=True
    )
    return prompt



def _regex_field(raw: str, key: str) -> str:
    m = re.search(rf'["\']{re.escape(key)}["\']\s*:\s*("([^"]*)"|\'([^\']*)\'|([0-9]+(\.[0-9]+)?))', raw)
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
        v = float(m.group(1))
        return max(0.0, min(100.0, v))
    except Exception:
        return 0.0


def _extract_fields(obj: dict):
    ans = normalize_answerset(obj.get("answer", ""))
    rea = safe_str(obj.get("reason", "")) or "(reason not parsed)"
    conf = _to_confidence(obj.get("confidence", ""))
    ok = len(ans) > 0 or len(rea) > 0 or conf > 0
    return ans, rea, conf, ok


def parse_json_from_text(text: str):
    raw = safe_str(text)

    try:
        obj = json.loads(raw)
        return _extract_fields(obj)
    except Exception:
        pass

    l = raw.find("{")
    r = raw.rfind("}")
    if l != -1 and r != -1 and r > l:
        cand = raw[l:r+1]
        try:
            obj = json.loads(cand)
            return _extract_fields(obj)
        except Exception:
            ans = _regex_field(raw, "answer")
            rea = _regex_field(raw, "reason")
            conf = _regex_field(raw, "confidence")
            ans2 = normalize_answerset(ans)
            rea2 = safe_str(rea) or "(reason not parsed)"
            conf2 = _to_confidence(conf)
            ok = (len(ans2) > 0) or (len(rea2) > 0) or (conf2 > 0)
            return ans2, rea2, conf2, ok

    return "", "(Failed to parse model output as JSON)", 0.0, False



def onepass_generate(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
            output_scores=False
        )

    seq = out.sequences[0]
    gen_tokens = seq[input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def process_dataset(input_path, output_path, model, tokenizer, device):
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"Output: {os.path.basename(output_path)}")
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
        print(f"  [ERROR] Row count mismatch: input={len(input_df)}, output={len(df)}")
        print(f"  Please delete {output_path} and re-run.")
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

    for idx in tqdm(range(len(df)), desc=f"  DianJin-R1-32B ({lang})"):
        row = df.iloc[idx]
        if row_already_done(row):
            continue

        q = safe_str(row[col_q])
        a = safe_str(row[col_a])
        b = safe_str(row[col_b])
        c = safe_str(row[col_c])
        d = safe_str(row[col_d])
        gt = normalize_answerset(row[col_gt])

        prompt = build_prompt_chattemplated(tokenizer, q, a, b, c, d, is_chinese)
        after_text = onepass_generate(model, tokenizer, prompt, device)

        pred, reason, conf, ok = parse_json_from_text(after_text)

        if not ok:
            pred = ""
            conf = 0.0
            reason = "(Model output not in JSON format, see Raw_Model_Output)"

        correct = (pred == gt) if gt else False

        df.at[idx, "Pred_Choice"] = pred
        df.at[idx, "Why_Choice"] = reason
        df.at[idx, "Is_Correct"] = bool(correct)
        df.at[idx, "Confidence"] = float(conf)
        df.at[idx, "Raw_Model_Output"] = after_text

        save_df_atomic(df, output_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = len(df)
    correct_count = df["Is_Correct"].sum()
    accuracy = correct_count / total * 100 if total > 0 else 0
    print(f"\n  Completed: {os.path.basename(input_path)}")
    print(f"  Accuracy: {correct_count}/{total} = {accuracy:.2f}%")


def main():
    print("="*60)
    print("DianJin-R1-32B Batch MCQ Evaluation (Normal)")
    print("Using model's built-in chat_template")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading model from: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )

    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "Tokenizer chat_template is empty/missing.\n"
            "DianJin-R1-32B (Qwen-based) should have a built-in chat_template."
        )
    print("  Chat template: OK (using model's built-in template)")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    model.eval()
    print("  Model loaded successfully")

    device = model.device if hasattr(model, "device") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  Device: {device}")

    for input_path in INPUT_PATHS:
        if not os.path.exists(input_path):
            print(f"\n[WARNING] File not found: {input_path}")
            continue
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{basename}_dianjin_r1_32b.csv")
        process_dataset(input_path, output_path, model, tokenizer, device)

    print("\n" + "="*60)
    print("All datasets processed!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
