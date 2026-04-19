
import os
import re
import json
import torch
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = "./model/CFGPT2-7B"
MODEL_NAME = "cfgpt2"
OUTPUT_DIR = "./outputs_cfgpt2"

TASKS = [
    {"input": "./missing_questions_Chinese.csv", "lang": "zh"},
    {"input": "./missing_questions_English.csv", "lang": "en"},
    {"input": "./original_questions_Chinese.csv", "lang": "zh"},
    {"input": "./original_questions_English.csv", "lang": "en"},
]

MAX_NEW_TOKENS = 1024



SYSTEM_PROMPT_ZH = "您是一位金融问题解决专家。您将收到一道单选或多选题。请返回一个严格的 JSON 数据，包含三个键：\"reason\" 和 \"answer\" 和 \"confidence\"。"

USER_PROMPT_ZH = """问题：
{question}

请严格按照以下格式输出 JSON 数据。请勿在 JSON 对象之外包含任何文本。
{{
  "reason": "简要解释这些选项正确的原因。",
  "answer": "仅输出大写字母 (A-Z)。对于多选题，请返回按字母顺序排列并连接起来的字母集合（例如，ACD）。对于单选题，请返回字母，例如 A 或 B 或 C 或 D。",
  "confidence": "你对本题你所回答的答案的信心程度是多少，在0-100里选一个。"
}}"""

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


def norm_answer(x):
    """统一用 [A-D](数据集只有 4 选项)"""
    if x is None:
        return ""
    s = str(x).strip().upper()
    letters = re.findall(r'[A-D]', s)
    return ''.join(sorted(set(letters)))


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

    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'latin1']:
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



def parse_response(text, lang="zh"):
    raw = safe_str(text)

    result_answer = ""
    result_reason = ""
    result_conf = 0.0

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            result_answer = norm_answer(obj.get("answer", ""))
            result_reason = safe_str(obj.get("reason", ""))
            result_conf = float(obj.get("confidence", 0))
            return result_answer, result_reason, min(100, max(0, result_conf))
    except Exception:
        pass

    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                result_answer = norm_answer(obj.get("answer", ""))
                result_reason = safe_str(obj.get("reason", ""))
                result_conf = float(obj.get("confidence", 0))
                return result_answer, result_reason, min(100, max(0, result_conf))
        except Exception:
            pass

    ans_match = re.search(r'"answer"\s*:\s*"([A-D]+)"', raw, flags=re.IGNORECASE)
    if ans_match:
        result_answer = ans_match.group(1).upper()
    else:
        ans_match = re.search(r'answer\s*[:：]\s*([A-D]+)', raw, flags=re.IGNORECASE)
        if ans_match:
            result_answer = ans_match.group(1).upper()
        else:
            matches = re.findall(r'\b([A-D])\b', raw)
            if matches:
                result_answer = matches[-1].upper()

    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw, flags=re.IGNORECASE)
    if reason_match:
        result_reason = reason_match.group(1)
    else:
        result_reason = raw[:500]

    conf_match = re.search(r'"confidence"\s*:\s*"?(\d+)"?', raw, flags=re.IGNORECASE)
    if conf_match:
        result_conf = float(conf_match.group(1))
        result_conf = max(0, min(100, result_conf))

    return norm_answer(result_answer), result_reason, result_conf



def build_question_text(row, lang="zh"):
    q = safe_str(row.get("Question", ""))
    a = safe_str(row.get("Option A", ""))
    b = safe_str(row.get("Option B", ""))
    c = safe_str(row.get("Option C", ""))
    d = safe_str(row.get("Option D", ""))

    if lang == "zh":
        return f"{q}\nA：{a}\nB：{b}\nC：{c}\nD：{d}"
    else:
        return f"{q}\nA: {a}\nB: {b}\nC: {c}\nD: {d}"


def build_prompt(row, lang="zh"):
    question_text = build_question_text(row, lang)
    if lang == "zh":
        system = SYSTEM_PROMPT_ZH
        user = USER_PROMPT_ZH.format(question=question_text)
    else:
        system = SYSTEM_PROMPT_EN
        user = USER_PROMPT_EN.format(question=question_text)
    return system, user



def infer_one(model, tokenizer, row, lang="zh"):
    system, user = build_prompt(row, lang)

    with torch.no_grad():
        response, _ = model.chat(
            tokenizer=tokenizer,
            query=user,
            history=[],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            meta_instruction=system
        )

    answer, reason, confidence = parse_response(response, lang)
    return {
        "answer": answer,
        "reason": reason,
        "confidence": confidence,
        "raw": response,
    }



def calculate_stats(df):
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0
    overall_total = 0

    gt_col = "Correct Answer" if "Correct Answer" in df.columns else "correct_answer"

    for _, row in df.iterrows():
        q_type = safe_str(row.get("Question_Type", "Unknown"))
        gt = norm_answer(row.get(gt_col, ""))
        pred = norm_answer(row.get("Pred_Choice", ""))

        if gt and pred:
            type_stats[q_type]["total"] += 1
            overall_total += 1
            if pred == gt:
                type_stats[q_type]["correct"] += 1
                overall_correct += 1

    return type_stats, overall_correct, overall_total


def print_report(type_stats, overall_correct, overall_total, name):
    print(f"\n{'='*70}")
    print(f"  {name} - Accuracy Report")
    print(f"{'='*70}")
    print(f"\n{'Question Type':<40} {'Correct':>8} {'Total':>8} {'Acc':>10}")
    print("-" * 70)

    for q_type, stats in sorted(type_stats.items(),
                                key=lambda x: x[1]["correct"]/max(x[1]["total"], 1),
                                reverse=True):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{q_type:<40} {stats['correct']:>8} {stats['total']:>8} {acc:>9.2f}%")

    overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print("-" * 70)
    print(f"{'OVERALL':<40} {overall_correct:>8} {overall_total:>8} {overall_acc:>9.2f}%")
    print("=" * 70)



def process_dataset(task, model, tokenizer):
    input_path = task["input"]
    lang = task["lang"]

    if not os.path.exists(input_path):
        log(f"[SKIP] Not found: {input_path}")
        return

    basename = os.path.basename(input_path).replace(".csv", "")
    output_path = os.path.join(OUTPUT_DIR, f"{basename}_{MODEL_NAME}.csv")

    log(f"Processing: {input_path}")
    log(f"Language: {'Chinese' if lang == 'zh' else 'English'}")

    df = read_file(input_path)
    total = len(df)
    log(f"Total: {total} rows")

    # 确保输出列存在
    for col in ["Pred_Choice", "Why_Choice", "Is_Correct", "Confidence", "Raw_Model_Output"]:
        if col not in df.columns:
            if col == "Is_Correct":
                df[col] = False
            elif col == "Confidence":
                df[col] = 0.0
            else:
                df[col] = ""

    gt_col = "Correct Answer" if "Correct Answer" in df.columns else "correct_answer"

    def row_done(row):
        pred = norm_answer(row.get("Pred_Choice", ""))
        reason = safe_str(row.get("Why_Choice", ""))
        # ERROR 情况下 norm_answer 会变空,所以不会被误判为 done
        return len(pred) > 0 and len(reason) > 0

    remaining = sum(1 for _, row in df.iterrows() if not row_done(row))
    log(f"Remaining: {remaining}/{total}")

    if remaining == 0:
        log("All done!")
        type_stats, overall_correct, overall_total = calculate_stats(df)
        print_report(type_stats, overall_correct, overall_total, basename)
        return

    for idx in tqdm(range(len(df)), desc=f"CFGPT2 ({lang})"):
        row = df.iloc[idx]
        if row_done(row):
            continue

        try:
            result = infer_one(model, tokenizer, row, lang)

            gt = norm_answer(row.get(gt_col, ""))
            pred = result["answer"]
            correct = (pred == gt) if gt else False

            df.at[idx, "Pred_Choice"] = pred
            df.at[idx, "Why_Choice"] = result["reason"]
            df.at[idx, "Is_Correct"] = correct
            df.at[idx, "Confidence"] = result["confidence"]
            df.at[idx, "Raw_Model_Output"] = result["raw"]

            q_type = safe_str(row.get("Question_Type", ""))[:20]
            mark = "✓" if correct else "✗"
            print(f"[{idx+1}/{total}] {q_type} | Pred={pred} | GT={gt} | Conf={result['confidence']:.0f} | {mark}")

        except Exception as e:
            log(f"[ERROR] Row {idx}: {e}")
            # 留空 Pred_Choice/Why_Choice,方便下次续跑
            df.at[idx, "Pred_Choice"] = ""
            df.at[idx, "Why_Choice"] = f"[ERROR] {str(e)[:200]}"
            df.at[idx, "Is_Correct"] = False
            df.at[idx, "Confidence"] = 0.0
            df.at[idx, "Raw_Model_Output"] = ""

        save_df_atomic(df, output_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"Saved: {output_path}")

    type_stats, overall_correct, overall_total = calculate_stats(df)
    print_report(type_stats, overall_correct, overall_total, basename)



def main():
    log("=" * 60)
    log("CFGPT2-7B MCQ Evaluation (using model.chat())")
    log("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log(f"Loading model: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    log("Model loaded!")

    for task in TASKS:
        process_dataset(task, model, tokenizer)

    log("All tasks completed!")


if __name__ == "__main__":
    main()
