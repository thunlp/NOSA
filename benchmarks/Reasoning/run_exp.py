import os
import sys
import subprocess

model_paths = {
    "8b_fullattn": "",
    "8b_infllmv2": "",
    "8b_nosa": "",
    
    "8b_arkvale_32": "",
    "8b_shadowkv_chunk_size=8": "",
    "8b_infllmv1_128": "",

    "8b_dma": "",
}

dataset_paths = {
    "math-500": "/somepath/Reasoning/math-500.jsonl",
    "gaokao": "DO_NOT_NEED", # 理科数学
    "gaokao-2": "DO_NOT_NEED", # 文科数学
    "gaokao-3": "DO_NOT_NEED", # 高考物理
}

def run(model:str, dataset:str, cuda_devices: list[int], run_posfix:str=""):
    # prepare directories
    save_dir = f"./results{run_posfix}/{dataset}/{model}"
    log_dir = f"./logs{run_posfix}/{dataset}/{model}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # prepare log files
    log_filename = f"{log_dir}/{dataset}-{model}-rank_0.log"
    f_log = open(log_filename, "w")

    cuda_devices_str = ",".join(map(str, cuda_devices))
    
    print(f"Launching {model} on {dataset} using GPUs: {cuda_devices_str}")

    cmd = [
        sys.executable, "-u", "test.py",
        "--model_type", model,
        "--data_type", dataset,
        "--model_path", model_paths[model],
        "--data_path", dataset_paths[dataset],
        "--save_path", save_dir,
        "--gen_len", "8192",
        "--batch_size", "4",
        "--cuda_devices", cuda_devices_str
    ]

    print(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, stdout=f_log, stderr=f_log)
    
    f_log.close()
    print(f"Finished {model} on {dataset}.\n")

def main():
    cuda_visible_divices = [0,1,2,3,4,5,6,7]

    for model in model_paths.keys():
        for dataset in dataset_paths.keys():
            run(model=model, dataset=dataset, cuda_devices=cuda_visible_divices, run_posfix="YOUR_OWN_POSFIX")

if __name__ == "__main__":
    main()
