# -*- coding: utf-8 -*-
"""
GPT-OSS MCQ Evaluation - Masked (NOTA) version
Supports both GPT-OSS-20B and GPT-OSS-120B (set MODEL_SIZE below).

Output columns:
- gpt_oss_{size}_reason
- gpt_oss_{size}_answer
- gpt_oss_{size}_confidence
- answer_from_reason
- original_correct_answer
- judge_by_answer
- judge_by_reason
- raw_output
"""

import os
import re
import json
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM



MODEL_SIZE = "20b"   # change to "120b" for the larger model

_MODEL_IDS = {
    "20b":  "openai/gpt-oss-20b",
    "120b": "openai/gpt-oss-120b",
}
assert MODEL_SIZE in _MODEL_IDS, f"MODEL_SIZE must be one of {list(_MODEL_IDS)}"
MODEL_ID = _MODEL_IDS[MODEL_SIZE]
MODEL_NAME = f"gpt_oss_{MODEL_SIZE}"

OUTPUT_DIR = os.path.abspath(f"./outputs_{MODEL_NAME}_masked")

# (masked_path, original_path_for_gold_lookup, lang)
FILE_CONFIGS = [
    ("./missing_questions_Chinese_masked.csv",
     "./missing_questions_Chinese.csv",
     "zh"),
    ("./missing_questions_English_masked.csv",
     "./missing_questions_English.csv",
     "en"),
]

COLUMN_MAPPINGS = {
    "Question": ["Question", "question"],
    "Option A": ["Option A", "option_a", "A"],
    "Option B": ["Option B", "option_b", "B"],
    "Option C": ["Option C", "option_c", "C"],
    "Option D": ["Option D", "option_d", "D"],
    "Correct Answer": ["Correct Answer", "correct_answer", "Answer", "answer"],
}

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



def log(msg):
    print(f"[{datetime.now()}] {msg}")


def safe_str(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def normalize_answerset(x):
    """Restricted to [A-D]; benchmark only has 4 options."""
    if x is None:
        return ""
    s = str(x).strip().upper()
    letters = re.findall(r"[A-D]", s)
    return "".join(sorted(set(letters)))


def read_file(path):
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
        if header[:4] == b'PK\x03\x04':
            return pd.read_excel(path)
    except Exception:
        pass
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="ignore")


def save_df_atomic(df, path):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def find_column(df, std):
    for v in COLUMN_MAPPINGS.get(std, []):
        if v in df.columns:
            return v
    return std if std in df.columns else None


def normalize_columns(df):
    col_map, missing = {}, []
    for std in COLUMN_MAPPINGS:
        actual = find_column(df, std)
        if actual:
            col_map[std] = actual
        else:
            missing.append(std)
    return col_map, missing



def extract_answer_from_reason(reason, lang="en"):
    """
    Heuristically extract an answer letter (A/B/C/D) from natural-language
    reasoning text. For GPT-OSS reasoning models, the 'analysis' channel
    often contains sentences like "The answer is A" or "选B" before the
    final JSON is emitted.
    """
    if not reason:
        return ""
    reason = reason.strip()

    if lang == "zh":
        zh_patterns = [
            r'(?:正确答案|答案)(?:是|应该是|为|应为)\s*[:：]?\s*([A-D])',
            r'(?:选择|选|答)\s*(?:选项)?\s*([A-D])',
            r'([A-D])\s*(?:是|为)\s*(?:正确|对|最佳)?(?:的)?(?:答案|选项|选择)',
            r'(?:选项|答案)\s*([A-D])\s*(?:正确|最佳|适当)',
            r'因此\s*[,，]?\s*(?:答案|选项|选)?\s*[:：]?\s*([A-D])',
            r'所以\s*[,，]?\s*(?:答案|选项|选)?\s*[:：]?\s*([A-D])',
        ]
        for pattern in zh_patterns:
            matches = re.findall(pattern, reason)
            if matches:
                return matches[-1].upper()

    en_patterns = [
        r'(?:the )?(?:correct )?answer (?:is|should be|would be)[:\s]*([A-D])',
        r'([A-D]) is (?:the )?(?:correct|right|best|most appropriate) (?:answer|option|choice)',
        r'(?:therefore|thus|hence|so)[,\s]*(?:the )?(?:answer|option)[:\s]*([A-D])',
        r'(?:select|choose|pick)[:\s]*(?:option )?([A-D])',
        r'option ([A-D]) is (?:correct|right|the answer)',
        r'([A-D]) (?:is|would be) (?:the )?(?:correct|appropriate|right)',
        r'(?:the )?(?:correct|right) (?:answer|option|choice) is ([A-D])',
        r'([A-D]) (?:provides|contains|addresses) (?:the )?(?:key|critical|essential|missing)',
    ]
    for pattern in en_patterns:
        matches = re.findall(pattern, reason, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Tail-based fallback: drop ranges like "A-D" / "A到D" then pick last option letter
    tail = reason[-200:] if len(reason) > 200 else reason
    for ep in [
        r'[A-D][、,][A-D][、,][A-D]',
        r'[A-D]\s*[-–—]\s*[A-D]',
        r'options?\s*[A-D]\s*(?:through|to)\s*[A-D]',
        r'选项\s*[A-D]\s*(?:到|至)\s*[A-D]',
    ]:
        tail = re.sub(ep, '', tail, flags=re.IGNORECASE)

    found = []
    for op in [
        r'(?:option)?\s*([A-D])(?:[\.:\)]|\s|$)',
        r'([A-D])(?:\.|\))',
        r'\(([A-D])\)',
    ]:
        found.extend(re.findall(op, tail, re.IGNORECASE))
    if found:
        return found[-1].upper()
    return ""



def _regex_field(raw, key):
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


def _to_confidence(x):
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


def _extract_fields(obj):
    ans = normalize_answerset(obj.get("answer", ""))
    rea = safe_str(obj.get("reason", "")) or "(reason not parsed)"
    conf = _to_confidence(obj.get("confidence", ""))
    ok = len(ans) > 0
    return ans, rea, conf, ok


def parse_json_from_text(text):
    """
    Harmony-aware JSON parser. GPT-OSS emits 'analysis' (CoT) and 'final'
    channels; the JSON answer sits in the final channel. After
    skip_special_tokens=True, channels appear as plain-text prefixes.

    (1) direct json.loads
    (2) last-{...} substring (works for both 'final'-only and 'analysis+final')
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



def build_prompt_chattemplated(tokenizer, row, col_map, lang):
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template")
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError("Tokenizer chat_template is empty/missing")

    q = safe_str(row[col_map["Question"]])
    a = safe_str(row[col_map["Option A"]])
    b = safe_str(row[col_map["Option B"]])
    c = safe_str(row[col_map["Option C"]])
    d = safe_str(row[col_map["Option D"]])

    if lang == "zh":
        question_block = f"{q}\n\nA：{a}\nB：{b}\nC：{c}\nD：{d}"
        system_msg = SYSTEM_PROMPT_CN
        user_msg = USER_PROMPT_TEMPLATE_CN.format(question=question_block)
    else:
        question_block = f"{q}\n\nA: {a}\nB: {b}\nC: {c}\nD: {d}"
        system_msg = SYSTEM_PROMPT_EN
        user_msg = USER_PROMPT_TEMPLATE_EN.format(question=question_block)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )



def load_done_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    done = set()
    try:
        df = pd.read_csv(output_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(output_path, encoding="utf-8", errors="ignore")
    if "ID" not in df.columns:
        return done
    for i in range(len(df)):
        rid = str(df.loc[i, "ID"]).strip()
        raw = safe_str(df.loc[i].get("raw_output", ""))
        reason = safe_str(df.loc[i].get(f"{MODEL_NAME}_reason", ""))
        if (raw or reason) and not reason.startswith("ERROR"):
            done.add(rid)
    return done



@torch.inference_mode()
def onepass_generate(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=REPETITION_PENALTY,
    )
    gen = out[0][input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()



def process_dataset(masked_path, original_path, lang, model, tokenizer, device):
    log("=" * 60)
    log(f"Masked:   {masked_path}")
    log(f"Original: {original_path}")
    log(f"Language: {'Chinese' if lang == 'zh' else 'English'}")
    log("=" * 60)

    df_masked = read_file(masked_path)
    col_map, missing = normalize_columns(df_masked)
    if missing:
        log(f"[ERROR] Missing columns in masked file: {missing}")
        return

    df_original = read_file(original_path)
    orig_col_map, orig_missing = normalize_columns(df_original)
    if orig_missing:
        log(f"[ERROR] Missing columns in original file: {orig_missing}")
        return

    if "ID" not in df_masked.columns:
        df_masked.insert(0, "ID", range(1, len(df_masked) + 1))
    if "ID" not in df_original.columns:
        df_original.insert(0, "ID", range(1, len(df_original) + 1))

    # ID -> original gold answer lookup
    original_answers = {}
    for i in range(len(df_original)):
        rid = str(df_original.loc[i, "ID"]).strip()
        original_answers[rid] = normalize_answerset(
            df_original.loc[i, orig_col_map["Correct Answer"]]
        )

    basename = os.path.basename(masked_path).replace(".csv", "")
    output_path = os.path.join(OUTPUT_DIR, f"{basename}_{MODEL_NAME}.csv")

    done_ids = load_done_ids(output_path)

    if os.path.exists(output_path):
        df = pd.read_csv(output_path, encoding="utf-8-sig")
    else:
        df = df_masked.copy()

    required_cols = [
        f"{MODEL_NAME}_reason",
        f"{MODEL_NAME}_answer",
        f"{MODEL_NAME}_confidence",
        "answer_from_reason",
        "original_correct_answer",
        "judge_by_answer",
        "judge_by_reason",
        "raw_output",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 if col == f"{MODEL_NAME}_confidence" else ""

    text_cols = [c for c in required_cols if c != f"{MODEL_NAME}_confidence"]
    for col in text_cols:
        df[col] = df[col].astype("object")
    df[f"{MODEL_NAME}_confidence"] = pd.to_numeric(
        df[f"{MODEL_NAME}_confidence"], errors="coerce"
    ).fillna(0.0)

    total = len(df)
    todo = sum(1 for i in range(total) if str(df.loc[i, "ID"]).strip() not in done_ids)

    if todo == 0:
        log("All questions already processed!")
        correct_by_answer = sum(1 for i in range(total) if df.loc[i].get("judge_by_answer") == "Correct")
        correct_by_reason = sum(1 for i in range(total) if df.loc[i].get("judge_by_reason") == "Correct")
        reason_extracted = sum(1 for i in range(total) if safe_str(df.loc[i].get("answer_from_reason", "")))
        log(f"\n[Method 1] Judge by {MODEL_NAME}_answer:")
        log(f"  Correct: {correct_by_answer} / {total} = {100*correct_by_answer/total:.2f}%")
        log(f"\n[Method 2] Judge by answer_from_reason:")
        log(f"  Extracted: {reason_extracted} / {total} = {100*reason_extracted/total:.2f}%")
        log(f"  Correct:   {correct_by_reason} / {total} = {100*correct_by_reason/total:.2f}%")
        return

    log(f"Total: {total}, Todo: {todo}, Done: {total - todo}")

    correct_by_answer = 0
    correct_by_reason = 0
    reason_extracted_count = 0
    processed = 0

    for idx in tqdm(range(total), desc=f"GPT-OSS-{MODEL_SIZE} Masked [{lang}]"):
        rid = str(df.loc[idx, "ID"]).strip()
        if rid in done_ids:
            if df.loc[idx].get("judge_by_answer") == "Correct":
                correct_by_answer += 1
            if df.loc[idx].get("judge_by_reason") == "Correct":
                correct_by_reason += 1
            if safe_str(df.loc[idx].get("answer_from_reason", "")):
                reason_extracted_count += 1
            processed += 1
            continue

        row = df.iloc[idx]
        gold_original = original_answers.get(rid, "")

        try:
            prompt = build_prompt_chattemplated(tokenizer, row, col_map, lang)
            raw = onepass_generate(model, tokenizer, prompt, device)
            pred_answer, reason, conf, _ = parse_json_from_text(raw)
            pred_from_reason = extract_answer_from_reason(reason, lang)

            judge_by_answer = "Correct" if pred_answer == gold_original else "Wrong"
            judge_by_reason = "Correct" if pred_from_reason == gold_original else "Wrong"

            df.at[idx, f"{MODEL_NAME}_reason"] = reason
            df.at[idx, f"{MODEL_NAME}_answer"] = pred_answer
            df.at[idx, f"{MODEL_NAME}_confidence"] = float(conf)
            df.at[idx, "answer_from_reason"] = pred_from_reason
            df.at[idx, "original_correct_answer"] = gold_original
            df.at[idx, "judge_by_answer"] = judge_by_answer
            df.at[idx, "judge_by_reason"] = judge_by_reason
            df.at[idx, "raw_output"] = raw

            processed += 1
            if judge_by_answer == "Correct":
                correct_by_answer += 1
            if judge_by_reason == "Correct":
                correct_by_reason += 1
            if pred_from_reason:
                reason_extracted_count += 1

            mark_ans = "✓" if judge_by_answer == "Correct" else "✗"
            mark_rsn = "✓" if judge_by_reason == "Correct" else "✗"
            print(f"[{idx+1}/{total}] ID={rid} | ans={pred_answer}({mark_ans}) "
                  f"rsn={pred_from_reason}({mark_rsn}) gold={gold_original} conf={conf:.0f}")

        except Exception as e:
            err = str(e)
            log(f"[ERROR] row {idx} (ID={rid}): {err[:150]}")
            df.at[idx, f"{MODEL_NAME}_reason"] = f"ERROR: {err[:300]}"
            df.at[idx, f"{MODEL_NAME}_answer"] = ""
            df.at[idx, f"{MODEL_NAME}_confidence"] = 0.0
            df.at[idx, "answer_from_reason"] = ""
            df.at[idx, "original_correct_answer"] = gold_original
            df.at[idx, "judge_by_answer"] = "Wrong"
            df.at[idx, "judge_by_reason"] = "Wrong"
            df.at[idx, "raw_output"] = ""
            processed += 1
            if "CUDA" in err or "out of memory" in err.lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_df_atomic(df, output_path)

    log(f"\nSaved: {output_path}")
    log("=" * 60)
    log(f"Processed: {processed}")
    log("")
    log(f"[Method 1] Judge by {MODEL_NAME}_answer:")
    acc1 = 100 * correct_by_answer / processed if processed > 0 else 0
    log(f"  Correct: {correct_by_answer} / {processed} = {acc1:.2f}%")
    log("")
    log(f"[Method 2] Judge by answer_from_reason:")
    extract_rate = 100 * reason_extracted_count / processed if processed > 0 else 0
    acc2 = 100 * correct_by_reason / processed if processed > 0 else 0
    log(f"  Extracted from reason: {reason_extracted_count} / {processed} = {extract_rate:.2f}%")
    log(f"  Correct:               {correct_by_reason} / {processed} = {acc2:.2f}%")
    log("=" * 60)



def main():
    log("=" * 60)
    log(f"GPT-OSS-{MODEL_SIZE} MCQ Evaluation - Masked (NOTA)")
    log(f"Model: {MODEL_ID}")
    log(f"Decoding: do_sample={DO_SAMPLE}, rep_penalty={REPETITION_PENALTY}, "
        f"max_new_tokens={MAX_NEW_TOKENS}")
    log("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log(f"Loading tokenizer & model from: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "Tokenizer chat_template is empty/missing. "
            "GPT-OSS ships with a built-in Harmony chat template."
        )
    log("  Chat template: OK (Harmony, model's built-in)")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    log("  Model loaded successfully")

    device = model.device if hasattr(model, "device") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log(f"  Device: {device}")

    for masked_path, original_path, lang in FILE_CONFIGS:
        masked_path = os.path.abspath(masked_path)
        original_path = os.path.abspath(original_path)
        if not os.path.exists(masked_path):
            log(f"[WARNING] Masked file not found, skipping: {masked_path}")
            continue
        if not os.path.exists(original_path):
            log(f"[WARNING] Original file not found, skipping: {original_path}")
            continue
        process_dataset(masked_path, original_path, lang, model, tokenizer, device)

    log("All tasks completed.")
    log(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
