from openai import OpenAI, AsyncOpenAI
import json
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

MAX_CONCURRENCY = 10

YOUR_API = ""
YOUR_BASE_URL = ""
assert YOUR_API != "" and YOUR_BASE_URL != ""

async def evaluate_item(client, semaphore, prompt_template, data_entry):
    async with semaphore:
        generation, answer = data_entry['generation'], data_entry['answer']
        query = prompt_template.format(generation=generation, answer=answer)
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            api_ans = response.choices[0].message.content

            if api_ans and "yes" in api_ans.lower():
                return 1
            return 0
            
        except Exception as e:
            print(f"Request failed: {e}")
            return 0

async def check(model, dataset, world_size, posfix:str=""):
    path="/somepath/Reasoning/results{posfix}/{dataset}/{model}/{dataset}-{model}-8192-1.0-rank_{rank}.jsonl"

    prompt = """
    You are a judge. Determine whether the generation explicitly contains the answer.
    Only explicit mention counts — implicit reasoning, hints, or derivations do not count.
    The wording does not need to match exactly; judge based on semantic equivalence.
    Ignore correctness of reasoning; only check if the answer is stated anywhere in the generation.
    If the answer appears, even if surrounded by extra text, output "yes".
    Otherwise output "no".
    Output only "yes" or "no".

    Generation: {generation}
    Answer: {answer}
    """

    data = []
    for i in range(world_size):
        with open(path.format(posfix=posfix, dataset=dataset, model=model, rank=i), 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            data.append(json.loads(line))

    with open(f"./results{posfix}/{dataset}/{model}/result.log", "a") as f:
        print(f"validating data of {path}", file=f)
        print(f"total data: {len(data)}", file=f)

    client = AsyncOpenAI(api_key=f"{YOUR_API}", base_url=f"{YOUR_BASE_URL}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [evaluate_item(client, semaphore, prompt, de) for de in data]
    results = await tqdm_async.gather(*tasks)
    
    correct = sum(results)
    accuracy = correct / len(data) if data else 0
    
    with open(f"./results{posfix}/{dataset}/{model}/result.log", "a") as f:
        print(f"validating data of {path}", file=f)
        print(f"accuracy: {accuracy}", file=f)

MODELS_TO_CHECK = []
DATASETS = []
assert len(MODELS_TO_CHECK)>0 and len(DATASETS)>0

async def main():
    for model in MODELS_TO_CHECK:
        for dataset in DATASETS:
            print(f"for {model} and {dataset}")
            await check(model=model, dataset=dataset, world_size=1, run_posfix="YOUR_OWN_POSFIX")

if __name__ == "__main__":
    asyncio.run(main())
