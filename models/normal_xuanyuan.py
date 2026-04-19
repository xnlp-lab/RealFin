
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
    ("/root/autodl-tmp/missing_questions_Chinese.csv",  "/root/autodl-tmp/missing_questions_Chinese_xuanyuan.csv",  "zh"),
    ("/root/autodl-tmp/missing_questions_English.csv",  "/root/autodl-tmp/missing_questions_English_xuanyuan.csv",  "en"),
    ("/root/autodl-tmp/original_questions_Chinese.csv", "/root/autodl-tmp/original_questions_Chinese_xuanyuan.csv", "zh"),
    ("/root/autodl-tmp/original_questions_English.csv", "/root/autodl-tmp/original_questions_English_xuanyuan.csv", "en"),
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


def build_question_text(question: str, a: str, b: str, c: str, d: str) -> str:
    return f"{question}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}"


def build_messages(question: str, a: str, b: str, c: str, d: str, lang: str = "zh"):
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

    # 1) 直接解析
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return (str(obj.get("reason", "")).strip(),
                    normalize_answerset(str(obj.get("answer", ""))),
                    to_confidence(obj.get("confidence", 50)))
    except Exception:
        pass

    # 2) 去掉 markdown 代码块
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

    # 3) 提取 {...}
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

    # 4) 正则兜底
    ans = ""
    reason = ""
    conf = 50
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
        raise RuntimeError("tokenizer.chat_template 不存在/为空，无法只用 chat-template。")

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


def process_one_file(input_path, output_path, lang, model, tokenizer, device_for_input):
    print(f"\n{'='*60}")
    print(f"处理文件: {input_path}")
    print(f"输出到: {output_path}")
    print(f"语言: {lang}")
    print(f"{'='*60}")

    df = read_table_auto(input_path)
    df = normalize_columns(df)

    required_cols = ["ID", "Question", "Option A", "Option B", "Option C", "Option D", "Correct Answer"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️ 跳过此文件，缺少列：{missing}")
        return

    done_ids = load_done_ids(output_path)
    out_fieldnames = list(df.columns) + ["xuanyuan_reason", "xuanyuan_answer", "xuanyuan_confidence", "xuanyuan_judge"]

    total = len(df)
    todo = sum(1 for i in range(total) if str(df.loc[i, "ID"]).strip() not in done_ids)

    if todo == 0:
        print(f"✅ 此文件已全部完成，跳过")
        return
    print(f"总题数: {total}, 待处理: {todo}, 已完成: {total - todo}")

    pbar = tqdm(total=todo, desc=f"[{lang.upper()}] Solving", ncols=120)

    for i in range(total):
        rid = str(df.loc[i, "ID"]).strip()
        if rid in done_ids:
            continue

        row = df.loc[i]
        q = str(row["Question"])
        a = str(row["Option A"])
        b = str(row["Option B"])
        c = str(row["Option C"])
        d = str(row["Option D"])
        gold = normalize_answerset(str(row["Correct Answer"]))

        messages = build_messages(q, a, b, c, d, lang)

        try:
            gen = infer_one(model, tokenizer, device_for_input, messages)
            reason, pred, confidence = parse_json_output(gen)
            judge = "Correct" if pred == gold else "Wrong"
        except Exception as e:
            reason = f"[ERROR] {repr(e)}"
            pred = ""
            confidence = -1
            judge = "Wrong"

        out_row = row.to_dict()
        out_row["xuanyuan_reason"] = reason
        out_row["xuanyuan_answer"] = pred
        out_row["xuanyuan_confidence"] = confidence
        out_row["xuanyuan_judge"] = judge

        append_one_row(output_path, out_fieldnames, out_row)

        pbar.update(1)
        pbar.set_postfix_str(f"ID={rid} pred={pred} gold={gold} conf={confidence} {judge}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    print(f"✅ 文件处理完成: {output_path}")


def main():
    print("加载模型中...")

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        use_fast=False,
        local_files_only=True,
        trust_remote_code=True
    )

    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "只用 chat-template，但当前 tokenizer.chat_template 为空/不存在。\n"
            "请检查模型/分词器或在 tokenizer_config.json 中加入 chat_template。"
        )

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

    for input_path, output_path, lang in FILE_CONFIGS:
        if not os.path.exists(input_path):
            print(f"⚠️ 文件不存在，跳过: {input_path}")
            continue
        process_one_file(input_path, output_path, lang, model, tokenizer, device_for_input)

    print(f"\n{'='*60}")
    print("全部完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
