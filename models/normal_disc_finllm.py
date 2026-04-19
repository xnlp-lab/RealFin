# -*- coding: utf-8 -*-
"""
DISC-FinLLM MCQ Evaluation - Normal version

Output columns:
- Pred_Choice:       model's predicted answer letter(s), normalized to [A-D]
- Why_Choice:        parsed "reason" field from model JSON output
- Is_Correct:        boolean, Pred_Choice == Correct Answer
- Confidence:        float in [0, 100], parsed from "confidence" field
- Raw_Model_Output:  full verbatim model output for audit
"""

import os
import re
import json
import gc
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig



MODEL_DIR = os.path.abspath("./model/Go4miii/DISC-FinLLM")
MODEL_NAME = "disc_finllm"

INPUT_PATHS = [
    os.path.abspath("./missing_questions_Chinese.csv"),
    os.path.abspath("./missing_questions_English.csv"),
    os.path.abspath("./original_questions_Chinese.csv"),
    os.path.abspath("./original_questions_English.csv"),
]

OUTPUT_DIR = os.path.abspath("./outputs_disc_finllm")
OUTPUT_PREFIX = "questions_with_pred"

# Generation hyperparameters (frozen for reproducibility)
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.05
DO_SAMPLE = False  # greedy decoding -- deterministic

COLUMN_MAPPINGS = {
    "Question": ["Question", "question"],
    "Option A": ["Option A", "option_a", "A"],
    "Option B": ["Option B", "option_b", "B"],
    "Option C": ["Option C", "option_c", "C"],
    "Option D": ["Option D", "option_d", "D"],
    "Correct Answer": ["Correct Answer", "correct_answer", "Answer", "answer"],
}



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



def log(msg):
    print(f"[{datetime.now()}] {msg}")


def read_table(path: str) -> pd.DataFrame:
    """Read csv/xlsx file with encoding auto-detection."""
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
        if header[:4] == b'PK\x03\x04':
            log(f"Detected Excel format: {os.path.basename(path)}")
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
    Normalize an answer string to a sorted, deduplicated subset of {A, B, C, D}.
    Restricted to [A-D] because the benchmark only has 4 options (A/B/C/D);
    restricting to [A-D] avoids false-positive matches on incidental uppercase
    letters in natural text (e.g., 'ROE', 'GDP').
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
    # Enforce dtypes so downstream analysis is stable
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
    """A row is 'done' if it has a non-empty, non-ERROR Pred_Choice and Why_Choice."""
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
        v = float(m.group(1))
        return max(0.0, min(100.0, v))
    except Exception:
        return 0.0


def _extract_fields(obj: dict):
    ans = normalize_answerset(obj.get("answer", ""))
    rea = safe_str(obj.get("reason", "")) or "(not parsed)"
    conf = _to_confidence(obj.get("confidence", ""))
    ok = len(ans) > 0
    return ans, rea, conf, ok


def parse_json_from_text(text: str):
    """
    Three-stage parser:
      (1) direct json.loads
      (2) substring from first '{' to last '}'
      (3) regex fallback for the three keys
    Returns (answer, reason, confidence, ok).
    """
    raw = safe_str(text)

    # (1) direct parse
    try:
        return _extract_fields(json.loads(raw))
    except Exception:
        pass

    # (2) extract JSON substring
    l = raw.find("{")
    r = raw.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return _extract_fields(json.loads(raw[l:r + 1]))
        except Exception:
            pass

    # (3) regex fallback
    ans = normalize_answerset(_regex_field(raw, "answer"))
    rea = safe_str(_regex_field(raw, "reason")) or "(not parsed)"
    conf = _to_confidence(_regex_field(raw, "confidence"))
    ok = len(ans) > 0
    return ans, rea, conf, ok



def build_question_block(question, a, b, c, d, is_chinese: bool):
    """Format the question and its 4 options."""
    sep = "：" if is_chinese else ": "
    return f"{question}\n\nA{sep}{a}\nB{sep}{b}\nC{sep}{c}\nD{sep}{d}"



def chat_with_model(model, tokenizer, gen_config,
                    question, a, b, c, d, is_chinese: bool):
    """
    Call the model using its official chat() method.

    Baichuan-13B-Chat's chat() internally builds the prompt with
    <reserved_user> / <reserved_assistant> special tokens and correctly
    handles a leading system message (see `build_chat_input` in the model's
    modeling_baichuan.py). We therefore pass system and user as separate
    messages, following the intended design of the chat template.

    Determinism is enforced by passing an explicit GenerationConfig with
    do_sample=False (overriding the default model config which samples).
    """
    question_block = build_question_block(question, a, b, c, d, is_chinese)

    if is_chinese:
        system_prompt = SYSTEM_PROMPT_CN
        user_content = USER_PROMPT_TEMPLATE_CN.format(question=question_block)
    else:
        system_prompt = SYSTEM_PROMPT_EN
        user_content = USER_PROMPT_TEMPLATE_EN.format(question=question_block)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Primary path: model.chat() with explicit generation_config
    if hasattr(model, "chat"):
        try:
            response = model.chat(tokenizer, messages, generation_config=gen_config)
            return response
        except TypeError:
            # Some older Baichuan checkpoints don't accept generation_config kwarg
            # -> fall through to manual build_chat_input path
            log("model.chat() does not accept generation_config; falling back.")

    # Fallback: manually call build_chat_input + generate (preserves template)
    if hasattr(model, "build_chat_input"):
        input_ids = model.build_chat_input(tokenizer, messages,
                                           max_new_tokens=MAX_NEW_TOKENS)
        input_ids = input_ids.to(model.device)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=gen_config,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = outputs[0][input_len:]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    raise RuntimeError(
        "Neither model.chat() nor build_chat_input() is available. "
        "This script requires a Baichuan-13B-Chat compatible checkpoint."
    )



def process_dataset(input_path: str, output_path: str,
                    model, tokenizer, gen_config):
    log(f"Processing: {os.path.basename(input_path)}")

    input_df = read_table(input_path)
    log(f"Loaded {len(input_df)} rows")

    col_map, missing = normalize_columns(input_df)
    if missing:
        log(f"[ERROR] Missing columns: {missing}")
        log(f"Available: {list(input_df.columns)}")
        return

    df = load_or_init_working_df(input_df, output_path)
    if len(df) != len(input_df):
        log(f"[ERROR] Row mismatch. Delete {output_path} and retry.")
        return

    is_chinese = is_chinese_dataset(input_path)
    lang = "Chinese" if is_chinese else "English"
    log(f"Language: {lang}")
    log(f"Total: {len(df)} questions")

    col_q = col_map["Question"]
    col_a = col_map["Option A"]
    col_b = col_map["Option B"]
    col_c = col_map["Option C"]
    col_d = col_map["Option D"]
    col_gt = col_map["Correct Answer"]

    correct_count = 0
    processed = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        if row_already_done(row):
            if bool(row.get("Is_Correct", False)):
                correct_count += 1
            processed += 1
            continue

        q = safe_str(row[col_q])
        a = safe_str(row[col_a])
        b = safe_str(row[col_b])
        c = safe_str(row[col_c])
        d = safe_str(row[col_d])
        gt = normalize_answerset(row[col_gt])

        try:
            response = chat_with_model(model, tokenizer, gen_config,
                                       q, a, b, c, d, is_chinese)
            pred, reason, conf, ok = parse_json_from_text(response)
            if not ok:
                pred = ""
                conf = 0.0
                reason = "(parse failed)"

            correct = (pred == gt) if gt else False

            df.at[idx, "Pred_Choice"] = pred
            df.at[idx, "Why_Choice"] = reason
            df.at[idx, "Is_Correct"] = bool(correct)
            df.at[idx, "Confidence"] = float(conf)
            df.at[idx, "Raw_Model_Output"] = response

            if correct:
                correct_count += 1
            processed += 1

            q_type = safe_str(row.get("Question_Type", ""))
            mark = "✓" if correct else "✗"
            print(f"[{idx+1}/{len(df)}] {q_type} | Pred={pred or '(empty)'} | GT={gt} | Conf={int(conf)} | {mark}")

        except Exception as e:
            err_msg = str(e)
            print(f"[{idx+1}/{len(df)}] ERROR: {err_msg[:150]}")
            df.at[idx, "Pred_Choice"] = ""
            df.at[idx, "Why_Choice"] = f"ERROR: {err_msg[:300]}"
            df.at[idx, "Is_Correct"] = False
            df.at[idx, "Confidence"] = 0.0
            df.at[idx, "Raw_Model_Output"] = ""
            if "CUDA" in err_msg or "out of memory" in err_msg.lower():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_df_atomic(df, output_path)

    accuracy = correct_count / processed * 100 if processed > 0 else 0
    log(f"Completed: {os.path.basename(input_path)}")
    log(f"Accuracy: {correct_count}/{processed} = {accuracy:.2f}%")
    print()



def main():
    log("=" * 60)
    log("DISC-FinLLM MCQ Evaluation (Normal)")
    log(f"Model: {MODEL_DIR}")
    log(f"Decoding: do_sample={DO_SAMPLE}, rep_penalty={REPETITION_PENALTY}, "
        f"max_new_tokens={MAX_NEW_TOKENS}")
    log("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("Loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=False,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Deterministic generation config (overrides model's default sampling config)
    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Some Baichuan-specific tokens (e.g., user/assistant ids) live in the
    # pretrained GenerationConfig; preserve those by merging.
    try:
        base_cfg = GenerationConfig.from_pretrained(MODEL_DIR)
        for attr in ("user_token_id", "assistant_token_id"):
            if hasattr(base_cfg, attr):
                setattr(gen_config, attr, getattr(base_cfg, attr))
    except Exception:
        pass

    model.eval()
    log("Model loaded.")

    for input_path in INPUT_PATHS:
        if not os.path.exists(input_path):
            log(f"[WARNING] Input not found, skipping: {input_path}")
            continue

        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(
            OUTPUT_DIR, f"{OUTPUT_PREFIX}_{basename}_{MODEL_NAME}.csv"
        )
        process_dataset(input_path, output_path, model, tokenizer, gen_config)

    log("All done.")
    log(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
