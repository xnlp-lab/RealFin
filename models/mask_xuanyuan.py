"""
XuanYuan3-70B MCQ Evaluation Mask version
  - judge_by_answer: 用模型 answer 字段与原始正确答案比较
  - judge_by_reason: 从 reason 中提取的答案与原始正确答案比较
"""

import os
import re
import csv
import json
import gc
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


FILE_CONFIGS = [
    # (masked_file, original_file, output_file, lang)
    ("/root/autodl-tmp/missing_questions_Chinese_masked.csv",
     "/root/autodl-tmp/missing_questions_Chinese.csv",
     "/root/autodl-tmp/missing_questions_Chinese_masked_xuanyuan.csv",
     "zh"),
    ("/root/autodl-tmp/missing_questions_English_masked.csv",
     "/root/autodl-tmp/missing_questions_English.csv",
     "/root/autodl-tmp/missing_questions_English_masked_xuanyuan.csv",
     "en"),
]

LOCAL_MODEL_PATH = "/root/autodl-tmp/model"
MAX_NEW_TOKENS = 1024
USE_8BIT = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



SYSTEM_PROMPT_ZH = "您是一位金融问题解决专家。您将收到一道单选或多选题。请返回一个严格的 JSON 数据，包含三个键：\"reason\" 和 \"answer\" 和 \"confidence\"。"

USER_PROMPT_TEMPLATE_ZH = """问题：
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



def read_table_auto(path: str) -> pd.DataFrame:
    with open(path, 'rb') as f:
        header = f.read(4)
    if header[:2] == b'PK':
        return pd.read_excel(path, engine='openpyxl')
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'Option_A': 'Option A', 'Option_B': 'Option B',
        'Option_C': 'Option C', 'Option_D': 'Option D',
        'Answer': 'Correct Answer', 'correct_answer': 'Correct Answer',
    }
    df = df.rename(columns=rename_map)
    if 'ID' not in df.columns:
        df.insert(0, 'ID', range(1, len(df) + 1))
    return df


def build_question_text(question, a, b, c, d):
    return f"{question}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}"


def build_messages(question, a, b, c, d, lang="zh"):
    question_text = build_question_text(question, a, b, c, d)
    if lang == "zh":
        system_prompt = SYSTEM_PROMPT_ZH
        user_prompt = USER_PROMPT_TEMPLATE_ZH.format(question=question_text)
    else:
        system_prompt = SYSTEM_PROMPT_EN
        user_prompt = USER_PROMPT_TEMPLATE_EN.format(question=question_text)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



def normalize_answerset(x: str) -> str:
    s = (x or "").strip().upper()
    letters = re.findall(r"[A-D]", s)
    if not letters:
        return ""
    return "".join(sorted(set(letters)))


def to_confidence(x) -> int:
    s = str(x).strip()
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return 50
    try:
        v = int(float(m.group(1)))
    except Exception:
        v = 50
    return max(0, min(100, v))


def parse_json_output(text: str):
    raw = (text or "").strip()
    if not raw:
        return "", "", 50

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return (str(obj.get("reason", "")).strip(),
                    normalize_answerset(str(obj.get("answer", ""))),
                    to_confidence(obj.get("confidence", 50)))
    except Exception:
        pass

    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return (str(obj.get("reason", "")).strip(),
                            normalize_answerset(str(obj.get("answer", ""))),
                            to_confidence(obj.get("confidence", 50)))
            except Exception:
                pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return (str(obj.get("reason", "")).strip(),
                        normalize_answerset(str(obj.get("answer", ""))),
                        to_confidence(obj.get("confidence", 50)))
        except Exception:
            pass

    ans, reason, conf = "", "", 50
    m_ans = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
    if m_ans:
        ans = normalize_answerset(m_ans.group(1))
    else:
        m_ans = re.search(r'"answer"\s*:\s*([A-D]+)', raw)
        if m_ans:
            ans = normalize_answerset(m_ans.group(1))
    m_reason = re.search(r'"reason"\s*:\s*"([\s\S]*?)"\s*,\s*"', raw)
    if m_reason:
        reason = m_reason.group(1).strip()
    m_conf = re.search(r'"confidence"\s*:\s*"?(\d+)"?', raw)
    if m_conf:
        conf = to_confidence(m_conf.group(1))
    if not reason:
        reason = raw[:500]
    return reason, ans, conf



def extract_answer_from_reason(reason: str, lang: str = "zh") -> str:
    if not reason:
        return ""
    reason = reason.strip()

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



def pick_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        prefer_keys = ["embed_tokens", "wte", "embeddings", "word_embeddings"]
        for k, v in model.hf_device_map.items():
            lk = str(k).lower()
            if any(pk in lk for pk in prefer_keys):
                if isinstance(v, str) and v.startswith("cuda"):
                    return torch.device(v)
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                return torch.device(v)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def infer_one(model, tokenizer, device_for_input: torch.device, messages) -> str:
    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError("tokenizer.chat_template 不存在/为空。")
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt")
    inputs = {k: v.to(device_for_input) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.05,
    )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen



def load_done_ids(output_path: str):
    if not os.path.exists(output_path):
        return set()
    done = set()
    try:
        with open(output_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "ID" not in (reader.fieldnames or []):
                return set()
            for row in reader:
                done.add(str(row.get("ID", "")).strip())
    except Exception:
        return set()
    return done


def append_one_row(output_path: str, fieldnames, row_dict: dict):
    file_exists = os.path.exists(output_path)
    need_header = (not file_exists) or (os.path.getsize(output_path) == 0)
    with open(output_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if need_header:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()
        os.fsync(f.fileno())



def process_one_file(masked_path, original_path, output_path, lang, model, tokenizer, device_for_input):
    print(f"\n{'='*60}")
    print(f"处理 Masked 文件: {masked_path}")
    print(f"原始文件 (获取真正答案): {original_path}")
    print(f"输出到: {output_path}")
    print(f"语言: {lang}")
    print(f"{'='*60}")

    df_masked = read_table_auto(masked_path)
    df_masked = normalize_columns(df_masked)

    df_original = read_table_auto(original_path)
    df_original = normalize_columns(df_original)

    original_answers = {}
    for i in range(len(df_original)):
        rid = str(df_original.loc[i, "ID"]).strip()
        ans = normalize_answerset(str(df_original.loc[i, "Correct Answer"]))
        original_answers[rid] = ans

    required_cols = ["ID", "Question", "Option A", "Option B", "Option C", "Option D", "Correct Answer"]
    missing = [c for c in required_cols if c not in df_masked.columns]
    if missing:
        print(f"⚠️ 跳过此文件，缺少列：{missing}")
        return

    done_ids = load_done_ids(output_path)

    out_fieldnames = list(df_masked.columns) + [
        "xuanyuan_reason",
        "xuanyuan_answer",
        "xuanyuan_confidence",
        "answer_from_reason",
        "original_correct_answer",
        "judge_by_answer",
        "judge_by_reason",
    ]

    total = len(df_masked)
    todo = sum(1 for i in range(total) if str(df_masked.loc[i, "ID"]).strip() not in done_ids)

    if todo == 0:
        print(f"✅ 此文件已全部完成，跳过")
        return
    print(f"总题数: {total}, 待处理: {todo}, 已完成: {total - todo}")

    correct_by_answer = 0
    correct_by_reason = 0
    reason_extracted_count = 0
    processed = 0

    pbar = tqdm(total=todo, desc=f"[{lang.upper()}] Solving", ncols=140)

    for i in range(total):
        rid = str(df_masked.loc[i, "ID"]).strip()
        if rid in done_ids:
            continue

        row = df_masked.loc[i]
        q = str(row["Question"])
        a = str(row["Option A"])
        b = str(row["Option B"])
        c = str(row["Option C"])
        d = str(row["Option D"])
        gold_original = original_answers.get(rid, "")

        messages = build_messages(q, a, b, c, d, lang)

        try:
            gen = infer_one(model, tokenizer, device_for_input, messages)
            reason, pred_answer, confidence = parse_json_output(gen)
            pred_from_reason = extract_answer_from_reason(reason, lang)
            judge_by_answer = "Correct" if pred_answer == gold_original else "Wrong"
            judge_by_reason = "Correct" if pred_from_reason == gold_original else "Wrong"
        except Exception as e:
            reason = f"[ERROR] {repr(e)}"
            pred_answer = ""
            pred_from_reason = ""
            confidence = -1
            judge_by_answer = "Wrong"
            judge_by_reason = "Wrong"

        processed += 1
        if judge_by_answer == "Correct":
            correct_by_answer += 1
        if judge_by_reason == "Correct":
            correct_by_reason += 1
        if pred_from_reason:
            reason_extracted_count += 1

        out_row = row.to_dict()
        out_row["xuanyuan_reason"] = reason
        out_row["xuanyuan_answer"] = pred_answer
        out_row["xuanyuan_confidence"] = confidence
        out_row["answer_from_reason"] = pred_from_reason
        out_row["original_correct_answer"] = gold_original
        out_row["judge_by_answer"] = judge_by_answer
        out_row["judge_by_reason"] = judge_by_reason
        append_one_row(output_path, out_fieldnames, out_row)

        pbar.update(1)
        pbar.set_postfix_str(f"ID={rid} ans={pred_answer} rsn={pred_from_reason} gold={gold_original}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()

    print(f"\n{'='*60}")
    print(f"文件处理完成: {output_path}")
    print(f"{'='*60}")
    print(f"处理题数: {processed}")
    print("")
    print("【方法 1】直接用 answer 字段判断:")
    print(f"  正确: {correct_by_answer} / {processed} = {100*correct_by_answer/processed:.2f}%")
    print("")
    print("【方法 2】从 reason 中提取答案判断:")
    print(f"  成功提取 reason 答案: {reason_extracted_count} / {processed} = {100*reason_extracted_count/processed:.2f}%")
    print(f"  正确: {correct_by_reason} / {processed} = {100*correct_by_reason/processed:.2f}%")
    print("=" * 60)


def main():
    print("加载模型中...")

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        use_fast=False,
        local_files_only=True,
        trust_remote_code=True
    )

    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError("tokenizer.chat_template 不存在/为空。")

    if USE_8BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )

    model.eval()
    device_for_input = pick_input_device(model)
    print(f"模型加载完成，输入设备: {device_for_input}")

    for masked_path, original_path, output_path, lang in FILE_CONFIGS:
        if not os.path.exists(masked_path):
            print(f"⚠️ Masked 文件不存在，跳过: {masked_path}")
            continue
        if not os.path.exists(original_path):
            print(f"⚠️ 原始文件不存在，跳过: {original_path}")
            continue
        process_one_file(masked_path, original_path, output_path, lang, model, tokenizer, device_for_input)

    print(f"\n{'='*60}")
    print("全部完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
