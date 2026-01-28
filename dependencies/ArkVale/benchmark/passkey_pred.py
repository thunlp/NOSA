import warnings

warnings.filterwarnings("once")
import os
import argparse
import random
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from numpy import random
import tqdm
import torch

# Define the model path and the corresponding prompt template
MODEL_CONFIGS = {
    "vicuna-v1.5-7b-16k": dict(
        path="lmsys/vicuna-7b-v1.5-16k",
        template="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Hello!\nASSISTANT: Hello!</s>\nUSER: {inst}\nASSISTANT:",
    ),
    "longchat-v1.5-7b-32k": dict(
        path="lmsys/longchat-7b-v1.5-32k",
        template="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{inst}[/INST]",
    ),
    "mistral-7b-inst": dict(
        path="mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "mistral-7b": dict(
        path="mistralai/Mistral-7B-v0.3",
    ),
    "llama-3-8b-inst-64k": dict(
        path="MaziyarPanahi/Llama-3-8B-Instruct-64k",
    ),
    "llama-3-8b-inst-1048k": dict(
        path="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    ),
}


def generate_prompt_landmark(n_garbage, loc):
    """Generates a text file and inserts an passkey at a random position."""

    n_garbage_prefix = int(n_garbage * loc)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = (
        f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    )
    final_question = "What is the pass key? The pass key is"
    lines = [task_description, garbage_prefix, information_line, garbage_suffix]
    return "\n".join(lines), final_question, str(pass_key)


def passkey_retrieval_test(
    args,
    model,
    tokenizer,
    device,
    n_garbage=10000,
    loc=0.5,
    use_3_stages_gen=False,
):
    from arkvale.adapter import generate

    prompt, prompt_postfix, answer = generate_prompt_landmark(n_garbage, loc)
    if use_3_stages_gen:
        generate.enable_3_stages_gen()
        q_input_ids = tokenizer(
            [prompt_postfix], truncation=False, return_tensors="pt"
        ).to(device)["input_ids"]
        generate.reset_q_input_ids(q_input_ids)
    else:
        prompt += prompt_postfix

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:]  # drop BOS
    # answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, :]  # drop BOS
    context_length = inputs.input_ids.shape[-1]
    max_new_tokens = answer_ids.shape[-1]
    if use_3_stages_gen:
        context_length += q_input_ids.shape[-1]
        max_new_tokens += q_input_ids.shape[-1]

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1e-9,
        pad_token_id=tokenizer.eos_token_id,
    )

    model_answer = output[0, -answer_ids.shape[-1] :].cpu()
    print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    is_correct = (model_answer == answer_ids[0]).all().item()
    print(
        f"The model output is '{tokenizer.decode(output[0, context_length:].cpu())}'. The model answer is '{tokenizer.decode(model_answer.cpu())}', is_correct : {is_correct}"
    )

    if use_3_stages_gen:
        generate.reset_q_input_ids()
        generate.disable_3_stages_gen()

    return is_correct, context_length


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="longchat-v1.5-7b-32k",
    )
    ap.add_argument("--name", type=str, default="default")
    ap.add_argument("--num-tests", type=int, default=20)
    # ap.add_argument(
    #     "--num-garbages", type=int, nargs="+", default=[38000, 76000, 114000]
    # )

    ap.add_argument("--arkvale", action="store_true", help="Enable ArkVale")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--page-budgets", type=int, default=128)
    ap.add_argument("--page-topks", type=int, default=32)
    ap.add_argument("--n-unlimited-layers", type=int, default=2)
    ap.add_argument("--n-max-bytes", type=int, default=40 * (1 << 30))
    ap.add_argument("--n-max-cpu-bytes", type=int, default=80 * (1 << 30))
    ap.add_argument("--use-3-stages-gen", action="store_true")

    args = ap.parse_args()
    if args.page_budgets < 0:
        args.page_budgets = None
    return args


def main():
    seed_everything(42)
    args = parse_args()

    # Define model config
    dev = torch.device("cuda")
    model_name: str = args.model
    path = MODEL_CONFIGS[model_name]["path"]
    num_tests = args.num_tests

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    if args.arkvale:
        from arkvale import adapter

        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map=dev
        ).eval()
        adapter.enable_arkvale(model, dtype=torch.float16, device=dev, **args.__dict__)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map=dev,
        ).eval()

    # This is a rough ratio to control the number of texts and tokens
    N_GARBAGES = [38000, 76000, 114000]
    if "N_GARBAGES" in os.environ:
        N_GARBAGES = [int(os.environ["N_GARBAGES"])]
    for n_garbage in N_GARBAGES:
        # 38000 76000 114000
        passed_tests = 0
        for i in tqdm.trange(num_tests):
            loc = i / num_tests
            is_correct, len_tokens = passkey_retrieval_test(
                args,
                model,
                tokenizer,
                dev,
                n_garbage=n_garbage,
                loc=loc,
                use_3_stages_gen=args.use_3_stages_gen,
            )
            passed_tests += is_correct

        accuracy = float(passed_tests) / num_tests

        with open("./passkey.jsonl", "a") as fp:
            json.dump(
                {
                    "model": model_name,
                    "name": args.name,
                    "length": len_tokens,
                    "accuracy": accuracy,
                },
                fp,
            )
            fp.write("\n")


if __name__ == "__main__":
    main()
