import argparse
import json
import os
import sys
import re
from tqdm import tqdm
import glob

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from model_utils import OpenAIModel

def parse_output(output, prefix="Answer:"):
    output = output.replace("\n", " ")

    def lstrip_string(s, sub):
        return re.sub(f'^{re.escape(sub)}', '', s, flags=re.IGNORECASE)
    patterns = [re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), re.compile(r"(?:^)(.*)(?:\n|$)")]
    for pat in patterns:
        matches = pat.search(output)
        if matches is not None:
            return lstrip_string(matches[1].strip(), prefix).strip() # 0 index includes the non-capturing group # lstrip again because for chat models sometimes it will repeat the prefix
    # if still not found, return None, but should actually never get this case...
    return None


# prompts inspired by https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG
judge_prompt = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.
Although you are not given the context, you will be given a set of correct answers that achieves full scores on all metrics, and you need to assess the provided answers using the correct answers.

Below is your grading rubric:

Fluency:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers.

Correctness:
- Score 0 (Incorrect): The answer does not agree with the provided correct answers at all.
- Score 1 (partly correct): Partly agree with one of the provided correct answers (for example, the question asks for a date and a person; the answer gets the date right but the person wrong).
- Score 2 (correct but not fully relevant): Fully agrees with one of the provided correct answers but mentions other completely irrelevant information. Note that extra details provided in the answer, even if not mentioned in the correct answers, should NOT be seen as irrelevant as long as they are relevant to the question to a reasonable extend.
- Score 3 (correct and relevant): Fully agrees with one of the provided correct answers and only provides information relevant to the question. Note that if the answer is longer than the correct answer, as long as everything in the answer is relevant to the question, it should still be given score 3. For example, if the correct answer is "the North Pole" and the answer is "They are headed for the North Pole", it should still be given a score of 3.

Now, read the following question, answer, and correct answers. First think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"fluency": 0, "correctness": 1}}.

Question: {question}
Correct answers: {correct_answers}
Answer: {parsed_output}
"""

# TODO
YOUR_OUTPUT_DIR = ""

def parse_json(text):
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if len(matches) > 0:
        try:
            r = json.loads(matches[-1])
        except:
            return None
        return r
    return None

def check_metrics(model, results_file, output_file):
    with open(results_file, "r") as f:
        results = json.load(f)

    sum_score = 0
    count_score = 0

    all_inputs = []
    for d in results["data"]:
        p = judge_prompt.format(question=d['question'], correct_answers=d['answer'], parsed_output=parse_output(d['output']))
        all_inputs.append(p)

    outputs = model.generate_batch(prompt=all_inputs, batch_file=output_file+".batch")
    for idx, o in enumerate(outputs):
        d = results["data"][idx]
        s = None

        if o is not None:
            scores = parse_json(o["output"])
            if scores is not None and "correctness" in scores and "fluency" in scores:
                s = scores
            else:
                print("Warning! Couldn't get a score")
                print(f"GPT-4 output: {o['output']}")

            if scores is not None:
                sum_score += scores["fluency"] * scores["correctness"]
                count_score += 1

        d["gpt-4-scores"] = s

        if idx < 10:
            print("=====================================")
            print(f"Prompt: {all_inputs[idx]}")
            print(f"Output: {o['output']}")
            print(f"Final score: {s}")

    results["averaged_metrics"]["gpt-4-score"] = sum_score / count_score
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    assert YOUR_OUTPUT_DIR != "", "Please set you output directory"

    model = OpenAIModel("gpt-4o-2024-05-13", temperature=0.1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--model_to_check", nargs="+", default=[])
    parser.add_argument("--tag", type=str, default="eval")
    args = parser.parse_args()
    num_shards = args.num_shards
    shard_idx = args.shard_idx

    if len(args.model_to_check) > 0:
        model_to_check = args.model_to_check
    else:
        # all models
        model_to_check = [
            "1b_nosa_sft",
            "3b_nosa_sft",
            "8b_nosa_sft",
            "1b_fullattn_sft",
            "3b_fullattn_sft",
            "8b_fullattn_sft",
            "1b_infllmv2_sft",
            "3b_infllmv2_sft",
            "8b_infllmv2_sft",
            "1b_infllmv1_64",
            "3b_infllmv1_64",
            "8b_infllmv1_64",
            "1b_infllmv1_128",
            "3b_infllmv1_128",
            "8b_infllmv1_128",
            "1b_shadowkv_chunk_size=8",
            "3b_shadowkv_chunk_size=8",
            "8b_shadowkv_chunk_size=8",
            "1b_shadowkv_chunk_size=64",
            "3b_shadowkv_chunk_size=64",
            "8b_shadowkv_chunk_size=64",
            "1b_arkvale_32",
            "3b_arkvale_32",
            "8b_arkvale_32",
            "8b_nosa_sft_pref",
            "8b_shadowkv_minf_chunk_size=8",
            "8b_shadowkv_minf_chunk_size=64",
            "8b_dma",
            "8b_dma_pref",
            "1b_dma",
            "1b_dma_pref",
            "3b_dma",
            "3b_dma_pref",
            "1b_nosa_pref",
            "3b_nosa_pref"
            "1b_shadowkv_minf_chunk_size=8",
            "1b_shadowkv_minf_chunk_size=64",
            "3b_shadowkv_minf_chunk_size=8",
            "3b_shadowkv_minf_chunk_size=64",
            "8b_nosa_sft_sb=8",
            "8b_nosa_sft_sb=16",
            "8b_nosa_sft_sb=24",
            "8b_nosa_sft_sb=32",
        ]

    all_paths = [glob.glob(f"{YOUR_OUTPUT_DIR}/{m}/longqa_16k/narrativeqa_*_{args.tag}_*.json") for m in model_to_check] \
        + [glob.glob(f"{YOUR_OUTPUT_DIR}/{m}/longqa_16k/infbench_*_{args.tag}_*.json") for m in model_to_check] \
        + [glob.glob(f"{YOUR_OUTPUT_DIR}/{m}/longqa_16k_s=20/narrativeqa_*_{args.tag}_*.json") for m in model_to_check] \
        + [glob.glob(f"{YOUR_OUTPUT_DIR}/{m}/longqa_16k_s=20/infbench_*_{args.tag}_*.json") for m in model_to_check]
    
    all_paths = [item for sublist in all_paths for item in sublist]
    all_paths = [p for p in all_paths if not os.path.exists(p.replace(".json", "-gpt4eval_o.json"))]
    all_paths = [p for p in all_paths if not p.endswith("-gpt4eval_o.json")]
    all_paths = all_paths[shard_idx::num_shards]
    print(f"Found {len(all_paths)} path")
    print(all_paths)

    for p in all_paths:
        newp = p.replace(".json", "-gpt4eval_o.json")
        print("evaluating path:", p)
        check_metrics(model, p, newp)
