# -*- coding: utf-8 -*-
"""
FinGPT (Qwen-7B + LoRA) MCQ Evaluation - Normal version

Output columns:
- fingpt_reason       : parsed "reason" field from model JSON (if any)
- fingpt_answer       : parsed answer letters, normalized to [A-D]
- fingpt_confidence   : parsed confidence in [0, 100]
- Is_Correct          : bool, fingpt_answer == gold
- raw_output          : full verbatim model output for audit
"""

import os
import re
import json
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "fingpt"

# Remote ids (downloaded on first run); can be replaced with local paths
BASE_MODEL = "Qwen/Qwen-7B"
LORA_MODEL = "FinGPT/fingpt-mt_qwen-7b_lora"

OUTPUT_DIR = "./outputs_fingpt"

# English only -- see module docstring for rationale
TASKS = [
    {"input": "./missing_questions_English.csv",  "lang": "en"},
    {"input": "./original_questions_English.csv", "lang": "en"},
]

# Generation hyperparameters (frozen for reproducibility)
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.05
DO_SAMPLE = False



SYSTEM_PROMPT_EN = "You are an expert in financial problem solving. You will be given a single- or multi-choice question. Please return STRICT JSON with three keys: \"reason\", \"answer\" and \"confidence\"."

USER_PROMPT_EN = """Question:
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
    """
    Restricted to [A-D] since the benchmark only has 4 options.
    Avoids false-positive matches on incidental uppercase letters
    (e.g., 'ROE', 'GDP', 'CFA') in natural-language text.
    """
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
    Three-stage parser. FinGPT rarely produces valid JSON, but we still
    attempt the same parsing path used for other models for consistency.
    """
    raw = safe_str(text)
    if not raw:
        return "", "(empty output)", 0.0, False

    # (1) direct parse
    try:
        return _extract_fields(json.loads(raw))
    except Exception:
        pass

    # (2) substring from first '{' to last '}'
    l = raw.find("{")
    r = raw.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return _extract_fields(json.loads(raw[l:r + 1]))
        except Exception:
            pass

    # (3) regex fallback on the three keys
    ans = normalize_answerset(_regex_field(raw, "answer"))
    rea = safe_str(_regex_field(raw, "reason")) or "(reason not parsed)"
    conf = _to_confidence(_regex_field(raw, "confidence"))
    ok = len(ans) > 0
    return ans, rea, conf, ok



def build_question_text(row):
    q = safe_str(row.get("Question", ""))
    a = safe_str(row.get("Option A", ""))
    b = safe_str(row.get("Option B", ""))
    c = safe_str(row.get("Option C", ""))
    d = safe_str(row.get("Option D", ""))
    return f"{q}\nA: {a}\nB: {b}\nC: {c}\nD: {d}"


def build_prompt(row):
    """
    Build FinGPT Instruction/Input/Answer prompt (English only).

    FinGPT does NOT use a chat template. The LoRA adapter was fine-tuned
    on the flat text format below; using apply_chat_template() would
    produce out-of-distribution inputs.
    """
    question_text = build_question_text(row)
    system = SYSTEM_PROMPT_EN
    user = USER_PROMPT_EN.format(question=question_text)
    return f"Instruction: {system}\nInput: {user}\nAnswer: "



@torch.inference_mode()
def infer_one(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()



def row_done(row):
    """Resumable: done iff fingpt_answer is a valid [A-D] set AND raw_output is non-empty."""
    ans = normalize_answerset(row.get("fingpt_answer", ""))
    raw = safe_str(row.get("raw_output", ""))
    # We accept non-empty raw_output even if answer is empty (parse-failed rows),
    # because running again will give the same result under deterministic decoding.
    return len(raw) > 0


def ensure_output_columns(df):
    for col, default in [
        ("fingpt_reason", ""),
        ("fingpt_answer", ""),
        ("fingpt_confidence", 0.0),
        ("Is_Correct", False),
        ("raw_output", ""),
    ]:
        if col not in df.columns:
            df[col] = default
    # Dtypes
    for c in ["fingpt_reason", "fingpt_answer", "raw_output"]:
        df[c] = df[c].astype("object")
    df["fingpt_confidence"] = pd.to_numeric(df["fingpt_confidence"], errors="coerce").fillna(0.0)
    df["Is_Correct"] = df["Is_Correct"].astype(bool)
    return df


def process_dataset(task, model, tokenizer, device):
    input_path = task["input"]
    if not os.path.exists(input_path):
        log(f"[SKIP] Not found: {input_path}")
        return

    basename = os.path.basename(input_path).replace(".csv", "")
    output_path = os.path.join(OUTPUT_DIR, f"results_{basename.replace('_questions', '')}_{MODEL_NAME}.csv")

    log(f"Processing: {input_path}")
    log(f"Output:     {output_path}")

    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(output_path, encoding="utf-8", errors="ignore")
    else:
        df = read_file(input_path)
    df = ensure_output_columns(df)

    gt_col = "Correct Answer" if "Correct Answer" in df.columns else \
             ("correct_answer" if "correct_answer" in df.columns else None)
    if gt_col is None:
        log(f"[ERROR] Missing Correct Answer column in {input_path}")
        return

    total = len(df)
    remaining = sum(1 for _, row in df.iterrows() if not row_done(row))
    log(f"Total: {total}  Remaining: {remaining}")
    if remaining == 0:
        log("All done.")
        correct = int(df["Is_Correct"].sum())
        log(f"Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
        return

    for idx in tqdm(range(total), desc=f"FinGPT (Qwen-7B+LoRA) [en]"):
        row = df.iloc[idx]
        if row_done(row):
            continue

        prompt = build_prompt(row)
        try:
            raw = infer_one(model, tokenizer, prompt, device)
            pred, reason, conf, _ = parse_json_from_text(raw)
            gt = normalize_answerset(row[gt_col])
            correct = (pred == gt) if (gt and pred) else False

            df.at[idx, "fingpt_reason"] = reason
            df.at[idx, "fingpt_answer"] = pred
            df.at[idx, "fingpt_confidence"] = float(conf)
            df.at[idx, "Is_Correct"] = bool(correct)
            df.at[idx, "raw_output"] = raw

            mark = "✓" if correct else "✗"
            qt = safe_str(row.get("Question_Type", ""))[:20]
            print(f"[{idx+1}/{total}] {qt} | Pred={pred or '(empty)'} | GT={gt} | {mark}")

        except Exception as e:
            err = str(e)
            log(f"[ERROR] row {idx}: {err[:150]}")
            df.at[idx, "fingpt_reason"] = f"ERROR: {err[:300]}"
            df.at[idx, "fingpt_answer"] = ""
            df.at[idx, "fingpt_confidence"] = 0.0
            df.at[idx, "Is_Correct"] = False
            df.at[idx, "raw_output"] = ""
            if "CUDA" in err or "out of memory" in err.lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_df_atomic(df, output_path)

    correct = int(df["Is_Correct"].sum())
    log(f"Completed: {os.path.basename(input_path)}")
    log(f"Accuracy: {correct}/{total} = {100*correct/total:.2f}%")


def main():
    log("=" * 60)
    log("FinGPT (Qwen-7B + LoRA) MCQ Evaluation (Normal, English only)")
    log(f"Base: {BASE_MODEL}")
    log(f"LoRA: {LORA_MODEL}")
    log(f"Decoding: do_sample={DO_SAMPLE}, rep_penalty={REPETITION_PENALTY}, "
        f"max_new_tokens={MAX_NEW_TOKENS}")
    log("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False,
    )
    # Qwen-7B does not set pad_token by default; required for generate()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    tokenizer.padding_side = "left"

    log("Loading base model ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    log(f"Applying LoRA adapter: {LORA_MODEL}")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL)
    model.eval()

    device = model.device if hasattr(model, "device") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log(f"Model ready on {device}")

    for task in TASKS:
        process_dataset(task, model, tokenizer, device)

    log("All done.")
    log(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
