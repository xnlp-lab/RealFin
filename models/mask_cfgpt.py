"""
function：
  1) 在 masked 数据集上运行模型（正确答案选项被改为"以上都不是"）
  2) 记录模型的 reason, answer, confidence
  3) 从 reason 中提取模型实际分析出的答案 (answer_from_reason)
  4) 用原始数据集的正确答案进行 judge
  5) 比较两种判断方式的 accuracy:
     - judge_by_answer: 直接用模型 answer 字段与原始答案比较
     - judge_by_reason: 用从 reason 中提取的答案与原始答案比较

output:
  - cfgpt2_reason
  - cfgpt2_answer
  - cfgpt2_confidence
  - answer_from_reason
  - original_correct_answer
  - judge_by_answer
  - judge_by_reason
  - raw_output
"""

import os
import re
import json
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = "./model/CFGPT2-7B"
MODEL_NAME = "cfgpt2"
OUTPUT_DIR = "./outputs_cfgpt2_masked"

FILE_CONFIGS = [
    # (masked_path, original_path, lang)
    ("./missing_questions_English_masked.csv",
     "./missing_questions_English.csv",
     "en"),
    ("./missing_questions_Chinese_masked.csv",
     "./missing_questions_Chinese.csv",
     "zh"),
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
    """规范化答案：只提取 A-D 字母，去重排序"""
    if x is None:
        return ""
    s = str(x).strip().upper()
    letters = re.findall(r'[A-D]', s)
    return ''.join(sorted(set(letters)))


def read_file(path):
    """读取 csv/excel，自动检测"""
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


def find_column(df, standard_name):
    if standard_name in COLUMN_MAPPINGS:
        for v in COLUMN_MAPPINGS[standard_name]:
            if v in df.columns:
                return v
    return standard_name if standard_name in df.columns else None


def normalize_columns(df):
    col_map = {}
    missing = []
    for std_name in COLUMN_MAPPINGS.keys():
        actual = find_column(df, std_name)
        if actual:
            col_map[std_name] = actual
        else:
            missing.append(std_name)
    return col_map, missing


def save_df_atomic(df, path):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)



def parse_response(text, lang="zh"):
    """解析模型输出 JSON，返回 (answer, reason, confidence)"""
    raw = safe_str(text)

    result_answer = ""
    result_reason = ""
    result_conf = 0.0

    # 1) 直接 JSON 解析
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            result_answer = norm_answer(obj.get("answer", ""))
            result_reason = safe_str(obj.get("reason", ""))
            result_conf = float(obj.get("confidence", 0))
            return result_answer, result_reason, min(100, max(0, result_conf))
    except Exception:
        pass

    # 2) 提取 {...} 部分
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

    # 3) 正则回退 - answer
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

    # 4) 正则回退 - reason
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw, flags=re.IGNORECASE)
    if reason_match:
        result_reason = reason_match.group(1)
    else:
        result_reason = raw[:500]

    # 5) 正则回退 - confidence
    conf_match = re.search(r'"confidence"\s*:\s*"?(\d+)"?', raw, flags=re.IGNORECASE)
    if conf_match:
        result_conf = float(conf_match.group(1))
        result_conf = max(0, min(100, result_conf))

    return norm_answer(result_answer), result_reason, result_conf


# =========================
# 从 reason 中提取答案 (与 mask_dianjin.py / mask.py 保持一致)
# =========================
def extract_answer_from_reason(reason: str, lang: str = "zh") -> str:
    """
    从 reason 文本中提取模型实际分析出的答案

    策略（按优先级）:
    1. 明确的结论性表述
    2. 肯定性表述
    3. 最后提到的选项（reason 结尾更可能是结论）
    """
    if not reason:
        return ""

    reason = reason.strip()

    # 中文模式
    zh_patterns = [
        r'(?:答案|正确答案|应选|应该选|选择|最终答案)[是为：:\s]*([A-D])',
        r'([A-D])[是为]?(?:正确|最佳|最合适|最恰当)(?:的)?(?:答案|选项|选择)',
        r'选([A-D])(?:选项)?(?:是|为)?(?:正确|最佳|最合适)',
        r'(?:因此|所以|综上|故)[，,]?(?:答案|选项|应选)?[是为：:\s]*([A-D])',
        r'([A-D])(?:选项)?[是为]正确的',
        r'([A-D])(?:选项)?(?:是|为)?(?:唯一|最)?正确',
        r'正确(?:的)?(?:答案|选项)[是为：:\s]*([A-D])',
        r'([A-D])(?:选项)?(?:提供了|包含了|涉及了)?关键信息',
        r'需要补充.*?([A-D])(?:选项)?',
    ]

    # 英文模式
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

    patterns = zh_patterns if lang == "zh" else en_patterns

    for pattern in patterns:
        matches = re.findall(pattern, reason, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # 兜底策略
    tail = reason[-200:] if len(reason) > 200 else reason

    exclude_patterns = [
        r'[A-D][、,][A-D][、,][A-D]',
        r'从[A-D]到[A-D]',
        r'[A-D]\s*[-–—]\s*[A-D]',
        r'options?\s*[A-D]\s*(?:through|to)\s*[A-D]',
    ]

    tail_clean = tail
    for ep in exclude_patterns:
        tail_clean = re.sub(ep, '', tail_clean, flags=re.IGNORECASE)

    option_patterns = [
        r'(?:选项|option)?\s*([A-D])(?:[\.、:：\)]|\s|$)',
        r'([A-D])(?:选项|\.|\)|）)',
        r'\(([A-D])\)',
    ]

    found_options = []
    for op in option_patterns:
        found_options.extend(re.findall(op, tail_clean, re.IGNORECASE))

    if found_options:
        return found_options[-1].upper()

    return ""



def build_question_text(q, a, b, c, d, lang="zh"):
    if lang == "zh":
        return f"{q}\nA：{a}\nB：{b}\nC：{c}\nD：{d}"
    else:
        return f"{q}\nA: {a}\nB: {b}\nC: {c}\nD: {d}"


def build_prompt(q, a, b, c, d, lang="zh"):
    question_text = build_question_text(q, a, b, c, d, lang)

    if lang == "zh":
        system = SYSTEM_PROMPT_ZH
        user = USER_PROMPT_ZH.format(question=question_text)
    else:
        system = SYSTEM_PROMPT_EN
        user = USER_PROMPT_EN.format(question=question_text)

    return system, user



def infer_one(model, tokenizer, q, a, b, c, d, lang="zh"):
    """使用 model.chat() 进行推理 - 按 CFGPT2 官方示例"""
    system, user = build_prompt(q, a, b, c, d, lang)

    with torch.no_grad():
        response, _ = model.chat(
            tokenizer=tokenizer,
            query=user,
            history=[],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,  # 与其它脚本对齐
            meta_instruction=system
        )

    answer, reason, confidence = parse_response(response, lang)
    return {
        "answer": answer,
        "reason": reason,
        "confidence": confidence,
        "raw": response,
    }



def load_done_ids(output_path: str) -> set:
    """按 ID 判定已完成的行（非 ERROR 才算）"""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path, encoding="utf-8-sig")
        done = set()
        for idx in range(len(df)):
            reason = safe_str(df.loc[idx].get(f"{MODEL_NAME}_reason", ""))
            if reason and not reason.startswith("ERROR"):
                done.add(str(df.loc[idx, "ID"]).strip())
        return done
    except Exception:
        return set()



def process_dataset(masked_path, original_path, lang, model, tokenizer):
    log("=" * 60)
    log(f"Processing masked: {masked_path}")
    log(f"Original (for gold answer): {original_path}")
    log(f"Language: {'Chinese' if lang == 'zh' else 'English'}")
    log("=" * 60)

    # 输出文件名
    basename = os.path.splitext(os.path.basename(masked_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{basename}_{MODEL_NAME}.csv")

    # 读 masked 数据
    df_masked = read_file(masked_path)
    col_map, missing = normalize_columns(df_masked)
    if missing:
        log(f"[ERROR] Missing columns in masked file: {missing}")
        log(f"Available: {list(df_masked.columns)}")
        return
    log(f"Column mapping: {col_map}")

    # 读 original 数据，建 ID -> original answer 的映射
    df_original = read_file(original_path)
    orig_col_map, orig_missing = normalize_columns(df_original)
    if orig_missing:
        log(f"[ERROR] Missing columns in original file: {orig_missing}")
        return

    # ID 列在 masked 文件里需要存在
    if "ID" not in df_masked.columns:
        df_masked.insert(0, "ID", range(1, len(df_masked) + 1))
    if "ID" not in df_original.columns:
        df_original.insert(0, "ID", range(1, len(df_original) + 1))

    original_answers = {}
    for i in range(len(df_original)):
        rid = str(df_original.loc[i, "ID"]).strip()
        ans = norm_answer(df_original.loc[i, orig_col_map["Correct Answer"]])
        original_answers[rid] = ans

    # 载入或初始化输出
    done_ids = load_done_ids(output_path)
    if os.path.exists(output_path):
        df = pd.read_csv(output_path, encoding="utf-8-sig")
    else:
        df = df_masked.copy()

    # 确保输出列存在
    required_out_cols = [
        f"{MODEL_NAME}_reason",
        f"{MODEL_NAME}_answer",
        f"{MODEL_NAME}_confidence",
        "answer_from_reason",
        "original_correct_answer",
        "judge_by_answer",
        "judge_by_reason",
        "raw_output",
    ]
    for col in required_out_cols:
        if col not in df.columns:
            if col == f"{MODEL_NAME}_confidence":
                df[col] = 0.0
            else:
                df[col] = ""

    # 强制文本列为 object 类型，避免被 pandas 识别成 float64
    text_cols = [
        f"{MODEL_NAME}_reason",
        f"{MODEL_NAME}_answer",
        "answer_from_reason",
        "original_correct_answer",
        "judge_by_answer",
        "judge_by_reason",
        "raw_output",
    ]
    for col in text_cols:
        df[col] = df[col].astype("object")

    # confidence 列保持数值型
    df[f"{MODEL_NAME}_confidence"] = pd.to_numeric(
        df[f"{MODEL_NAME}_confidence"], errors="coerce"
    ).fillna(0.0)

    # 取列名
    col_q = col_map["Question"]
    col_a = col_map["Option A"]
    col_b = col_map["Option B"]
    col_c = col_map["Option C"]
    col_d = col_map["Option D"]

    total = len(df)
    todo = sum(1 for i in range(total) if str(df.loc[i, "ID"]).strip() not in done_ids)

    if todo == 0:
        log("All questions already processed!")
        correct_by_answer = sum(1 for i in range(total) if df.loc[i].get("judge_by_answer") == "Correct")
        correct_by_reason = sum(1 for i in range(total) if df.loc[i].get("judge_by_reason") == "Correct")
        reason_extracted = sum(1 for i in range(total) if df.loc[i].get("answer_from_reason"))

        log(f"\n【Method 1】Judge by {MODEL_NAME}_answer:")
        log(f"  Correct: {correct_by_answer} / {total} = {100*correct_by_answer/total:.2f}%")
        log(f"\n【Method 2】Judge by answer_from_reason:")
        log(f"  Extracted: {reason_extracted} / {total} = {100*reason_extracted/total:.2f}%")
        log(f"  Correct: {correct_by_reason} / {total} = {100*correct_by_reason/total:.2f}%")
        return

    log(f"Total: {total}, Todo: {todo}, Done: {total - todo}")

    # 统计
    correct_by_answer = 0
    correct_by_reason = 0
    reason_extracted_count = 0
    processed = 0

    for idx in tqdm(range(len(df)), desc=f"CFGPT2 Masked ({lang})"):
        rid = str(df.loc[idx, "ID"]).strip()

        # 已完成的行：累加统计，跳过推理
        if rid in done_ids:
            if df.loc[idx].get("judge_by_answer") == "Correct":
                correct_by_answer += 1
            if df.loc[idx].get("judge_by_reason") == "Correct":
                correct_by_reason += 1
            if df.loc[idx].get("answer_from_reason"):
                reason_extracted_count += 1
            processed += 1
            continue

        row = df.iloc[idx]
        q = safe_str(row[col_q])
        a = safe_str(row[col_a])
        b = safe_str(row[col_b])
        c = safe_str(row[col_c])
        d = safe_str(row[col_d])

        gold_original = original_answers.get(rid, "")

        try:
            result = infer_one(model, tokenizer, q, a, b, c, d, lang)
            pred_answer = result["answer"]
            reason = result["reason"]
            conf = result["confidence"]
            raw = result["raw"]

            # 从 reason 提取答案
            pred_from_reason = extract_answer_from_reason(reason, lang)

            # 判断
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

            # 累加统计
            processed += 1
            if judge_by_answer == "Correct":
                correct_by_answer += 1
            if judge_by_reason == "Correct":
                correct_by_reason += 1
            if pred_from_reason:
                reason_extracted_count += 1

            mark_ans = "✓" if judge_by_answer == "Correct" else "✗"
            mark_rsn = "✓" if judge_by_reason == "Correct" else "✗"
            print(f"[{idx+1}/{total}] ID={rid} | ans={pred_answer}({mark_ans}) rsn={pred_from_reason}({mark_rsn}) gold={gold_original} conf={conf:.0f}")

        except Exception as e:
            log(f"[ERROR] Row {idx} (ID={rid}): {e}")
            df.at[idx, f"{MODEL_NAME}_reason"] = f"ERROR: {str(e)[:200]}"
            df.at[idx, f"{MODEL_NAME}_answer"] = ""
            df.at[idx, f"{MODEL_NAME}_confidence"] = 0.0
            df.at[idx, "answer_from_reason"] = ""
            df.at[idx, "original_correct_answer"] = gold_original
            df.at[idx, "judge_by_answer"] = "Wrong"
            df.at[idx, "judge_by_reason"] = "Wrong"
            df.at[idx, "raw_output"] = ""
            processed += 1

        # 每行保存
        save_df_atomic(df, output_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nSaved: {output_path}")

    # 打印统计
    log("=" * 60)
    log(f"Completed: {output_path}")
    log("=" * 60)
    log(f"Processed: {processed}")
    log("")
    log(f"【Method 1】Judge by {MODEL_NAME}_answer:")
    acc1 = 100 * correct_by_answer / processed if processed > 0 else 0
    log(f"  Correct: {correct_by_answer} / {processed} = {acc1:.2f}%")
    log("")
    log(f"【Method 2】Judge by answer_from_reason:")
    extract_rate = 100 * reason_extracted_count / processed if processed > 0 else 0
    acc2 = 100 * correct_by_reason / processed if processed > 0 else 0
    log(f"  Extracted from reason: {reason_extracted_count} / {processed} = {extract_rate:.2f}%")
    log(f"  Correct: {correct_by_reason} / {processed} = {acc2:.2f}%")
    log("=" * 60)



def main():
    log("=" * 60)
    log("CFGPT2-7B MCQ Evaluation - Masked Version")
    log("Using model.chat() official interface")
    log("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log(f"Loading model: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    log("Model loaded!")

    for masked_path, original_path, lang in FILE_CONFIGS:
        masked_path = os.path.abspath(masked_path)
        original_path = os.path.abspath(original_path)

        if not os.path.exists(masked_path):
            log(f"[WARNING] Masked file not found: {masked_path}")
            continue
        if not os.path.exists(original_path):
            log(f"[WARNING] Original file not found: {original_path}")
            continue

        process_dataset(masked_path, original_path, lang, model, tokenizer)

    log("=" * 60)
    log("All tasks completed!")
    log(f"Outputs saved to: {OUTPUT_DIR}")
    log("=" * 60)


if __name__ == "__main__":
    main()
