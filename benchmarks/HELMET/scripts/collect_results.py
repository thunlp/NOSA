import os
import json
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm

SEEDS = [0, 42, 3407]

# TODO
YOUR_OUTPUT_DIR = ""

dataset_to_metrics = {
    "json_kv": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",
    
    "narrativeqa": ["gpt-4-score"],
    "msmarco_rerank_psg": "NDCG@10",
    
    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",
    
    "qmsum": "rougeL_recall",
    "multi_lexsum": ["gpt-4-f1"],
    
    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",
    
    "infbench_qa": ["rougeL_f1"],
    "infbench_choice": ["exact_match"],
    "infbench_sum": ["gpt-4-f1"],
    
    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}

dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}

custom_avgs = {
    "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall", "ruler_niah_mv ruler_recall"],
    "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match', 'triviaqa substring_exact_match',],
    "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match', 'nlu exact_match'],
    "Cite": ['alce_asqa str_em', 'alce_asqa citation_rec', 'alce_asqa citation_prec', 'alce_qampari qampari_rec_top5', 'alce_qampari citation_rec', 'alce_qampari citation_prec', ],
    "Re-rank": ['msmarco_rerank_psg NDCG@10', ],
    "LongQA": ['narrativeqa gpt-4-score', 'infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
    "Summ": ['infbench_sum gpt-4-f1', 'multi_lexsum gpt-4-f1', ],
    "Ours": ['Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
}

@dataclass
class arguments:
    tag: str = "v1"
    input_max_length: int = 131072
    generation_max_length: int = 100
    generation_min_length: int = 0
    max_test_samples: int = 100
    shots: int = 2
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_chat_template: bool = False
    seed: int = 42
    test_name: str = ""
    dataset: str = "nq"
    output_dir: str = "output"
    popularity_threshold: float = 3
    config: str = ""
    
    category: str = "synthetic"
    
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_path(self):
        tag = self.tag
        path = os.path.join(
            self.output_dir, self.config,
            "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}.json_seed{args.seed}.json".format(args=self, tag=tag)
        )

        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        if "alce" in self.dataset:
            return path + ".score"
        
        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None
    
    def get_averaged_metric(self):
        path = self.get_path()
        if not os.path.exists(path):
            print(f"path doesn't exist {path}")
            return None
        with open(path) as f:
            results = json.load(f)
        
        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}
        
        s = {m : v * (100 if m == "gpt-4-f1" else 1) * (100/3 if m == "gpt-4-score" else 1) for m, v in s.items()}
        print("found scores:", s)
        return s
        
    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')

        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []        
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d['ctxs'])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)
        
        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")
    
    def get_multi_seed_metrics(self, seeds=SEEDS):
        dsimple, metric_names = self.get_metric_name()
        if dsimple is None:
            return None

        scores_by_metric = {m: [] for m in metric_names}

        original_seed = self.seed

        for sd in seeds:
            self.seed = sd
            path = self.get_path()
            if not os.path.exists(path):
                print(f"path doesn't exist {path}")
                self.seed = original_seed
                return None

            with open(path) as f:
                try:
                    results = json.load(f)
                except Exception as e:
                    print(e)
                    print(path)
                    exit(1)

            if path.endswith(".score"):
                base = results
            else:
                base = results["averaged_metrics"]

            for m in metric_names:
                if m not in base:
                    print("metric doesn't exist", m, "in", path)
                    self.seed = original_seed
                    return None
                v = base[m]
                v = v * (100 if m == "gpt-4-f1" else 1) * (100/3 if m == "gpt-4-score" else 1)
                scores_by_metric[m].append(v)

        self.seed = original_seed
        return scores_by_metric

if __name__ == "__main__":
    assert YOUR_OUTPUT_DIR != "", "Please set you output directory"

    # comment out the models you don't want to include, or add the new ones 
    models_configs = [
        {"model": "1b_fullattn_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "1b_infllmv2_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "1b_nosa_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "1b_nosa_pref", "use_chat_template": False, "training_length": 16384},
        
        {"model": "3b_fullattn_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_infllmv2_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_nosa_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_nosa_pref", "use_chat_template": False, "training_length": 16384},
        
        {"model": "8b_fullattn_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_infllmv2_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_sft", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_pref", "use_chat_template": False, "training_length": 16384},

        {"model": "1b_shadowkv_chunk_size=8", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_shadowkv_chunk_size=8", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_shadowkv_chunk_size=8", "use_chat_template": False, "training_length": 16384},

        {"model": "1b_shadowkv_chunk_size=64", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_shadowkv_chunk_size=64", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_shadowkv_chunk_size=64", "use_chat_template": False, "training_length": 16384},

        {"model": "1b_infllmv1_64", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_infllmv1_64", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_infllmv1_64", "use_chat_template": False, "training_length": 16384},

        {"model": "1b_infllmv1_128", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_infllmv1_128", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_infllmv1_128", "use_chat_template": False, "training_length": 16384},

        {"model": "1b_arkvale_32", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_arkvale_32", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_arkvale_32", "use_chat_template": False, "training_length": 16384},

        {"model": "1b_shadowkv_minf_chunk_size=8", "use_chat_template": False, "training_length": 16384},
        {"model": "1b_shadowkv_minf_chunk_size=64", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_shadowkv_minf_chunk_size=8", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_shadowkv_minf_chunk_size=64", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_shadowkv_minf_chunk_size=8", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_shadowkv_minf_chunk_size=64", "use_chat_template": False, "training_length": 16384},

        {"model": "1b_dma", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_dma", "use_chat_template": False, "training_length": 16384},
        {"model": "1b_dma_pref", "use_chat_template": False, "training_length": 16384},
        {"model": "3b_dma_pref", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_dma", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_dma_pref", "use_chat_template": False, "training_length": 16384},

        {"model": "8b_nosa_sft_1_1", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_sft_1_3", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_sft_1_7", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_sft_3_1", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_sft_sb=8", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_sft_sb=16", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_sft_sb=24", "use_chat_template": False, "training_length": 16384},
        {"model": "8b_nosa_sft_sb=32", "use_chat_template": False, "training_length": 16384},
    ]

    # set your configs here, only include the ones that you ran
    config_files = [
        "configs/recall_16k.yaml",
        "configs/rag_16k.yaml",
        "configs/longqa_16k.yaml",
        "configs/summ_16k.yaml",
        "configs/rerank_16k.yaml",
        "configs/icl_16k.yaml",
        "configs/cite_16k.yaml",

        "configs/recall_16k_s=20.yaml",
        "configs/rag_16k_s=20.yaml",
        "configs/longqa_16k_s=20.yaml",
        "configs/summ_16k_s=20.yaml",
        "configs/rerank_16k_s=20.yaml",
        "configs/icl_16k_s=20.yaml",
        "configs/cite_16k_s=20.yaml",

        "configs/recall_32k.yaml",
        "configs/rag_32k.yaml",
        "configs/longqa_32k.yaml",
        "configs/summ_32k.yaml",
        "configs/rerank_32k.yaml",
        "configs/icl_32k.yaml",
        "configs/cite_32k.yaml",

        "configs/recall_64k.yaml",
        "configs/rag_64k.yaml",
        "configs/longqa_64k.yaml",
        "configs/summ_64k.yaml",
        "configs/rerank_64k.yaml",
        "configs/icl_64k.yaml",
        "configs/cite_64k.yaml",

        "configs/recall.yaml",
        "configs/rag.yaml",
        "configs/longqa.yaml",
        "configs/summ.yaml",
        "configs/rerank.yaml",
        "configs/icl.yaml",
        "configs/cite.yaml",
    ]

    dataset_configs = []
    for file in config_files:
        c = yaml.safe_load(open(file))
        
        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
        for d, t, l, g in zip(c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','), c['generation_max_length'].split(',')):
            dataset_configs.append({
                "config": file[8:-5],
                "dataset": d, 
                "test_name": os.path.basename(os.path.splitext(t)[0]), 
                "input_max_length": int(l), "generation_max_length": int(g), 
                "max_test_samples": c['max_test_samples'], 
                'use_chat_template': c['use_chat_template'], 
                'shots': c['shots']}
            )
    
    print(dataset_configs)    

    df = []
    failed_paths = []

    for model in tqdm(models_configs):
        args = arguments()
        args.tag = "eval"  # SET YOUR TAG HERE
        args.output_dir = f"{YOUR_OUTPUT_DIR}/{model['model']}"
    
        for dataset in dataset_configs:
            args.update(model)
            args.update(dataset)

            get_metric_name_return = args.get_metric_name()
            if(get_metric_name_return is None):
                continue
            dsimple, mnames = get_metric_name_return
            scores_by_metric = args.get_multi_seed_metrics(SEEDS)

            if scores_by_metric is None:
                failed_paths.append(args.get_path())
                continue

            for k in mnames:
                vals = np.array(scores_by_metric[k], dtype=float)
                row = {
                    **asdict(args),
                    **model,
                    "metric name": k,
                    
                    "metric_seed_0": float(vals[0]),
                    "metric_seed_42": float(vals[1]),
                    "metric_seed_3407": float(vals[2]),

                    "metric": float(vals.mean()),
                    "metric_std": float(vals.std()),
                    "dataset_simple": dsimple + " " + k,
                    "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}",
                }

                df.append(row)


        all_df = pd.DataFrame(df)
        # 1. raw data 直接全量 dump，保持你现在的行为
        with open("result_all_tmp.csv", "w") as f:
            print(all_df.to_csv(index=False), file=f)

        # ========== 从这里开始是新的聚合逻辑 ==========
        
        seed_col_map = {
            0: "metric_seed_0",
            42: "metric_seed_42",
            3407: "metric_seed_3407",
        }

        summary_rows = []

        # 每个 (input_max_length, model) 做一行
        group_cols = ["input_max_length", "model"]
        for (in_len, model_name), g in all_df.groupby(group_cols):
            # g：这个 model 在所有 dataset_simple 上的行集合
            # 先做一个 dict: dataset_simple -> {seed -> value}
            data_by_ds = {}
            for _, r in g.iterrows():
                ds = r["dataset_simple"]
                if ds not in data_by_ds:
                    data_by_ds[ds] = {s: None for s in SEEDS}
                for s in SEEDS:
                    col = seed_col_map[s]
                    data_by_ds[ds][s] = float(r[col])

            # 一个小工具函数：给定一组 dataset_simple key，算出
            #  对每个 seed 的“这个任务的宏平均值”
            def compute_macro_from_dataset_keys(dataset_keys):
                # 返回: {seed -> macro_mean_for_this_seed}
                per_seed_vals = {s: [] for s in SEEDS}
                for ds_key in dataset_keys:
                    if ds_key not in data_by_ds:
                        # 某些模型可能没跑某个 ds，直接跳过并给个提醒
                        print(f"[WARN] dataset_simple '{ds_key}' missing for model {model_name}")
                        continue
                    for s in SEEDS:
                        per_seed_vals[s].append(data_by_ds[ds_key][s])
                # 对每个 seed，把该任务涉及到的所有数据集的分数做平均
                macro_seed = {}
                for s in SEEDS:
                    vals = per_seed_vals[s]
                    if len(vals) == 0:
                        macro_seed[s] = float("nan")
                    else:
                        macro_seed[s] = float(np.mean(vals))
                return macro_seed

            row = {
                "model": model_name,
                "input_max_length": in_len,
            }

            # 先把 custom_avgs 中的“宏任务”取出来（去掉 "Ours"，Ours 单独算）
            macro_tasks = [k for k in custom_avgs.keys() if k != "Ours"]

            # 存一下每个宏任务在每个 seed 上的值，后面算 Ours 要用
            macro_seed_values = {}

            # 依次计算每个宏任务的 mean/std
            for macro in macro_tasks:
                ds_keys = custom_avgs[macro]  # 例如 ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", ...]
                seed2val = compute_macro_from_dataset_keys(ds_keys)
                macro_seed_values[macro] = seed2val

                vals = np.array([seed2val[s] for s in SEEDS], dtype=float)
                row[f"{macro}_mean"] = float(np.nanmean(vals))
                row[f"{macro}_std"] = float(np.nanstd(vals))

            # 现在算 Ours：
            #  对每个 seed，先对所有宏任务的 macro_mean 做一次平均
            ours_seed = {}
            for s in SEEDS:
                vals = np.array([macro_seed_values[m][s] for m in macro_tasks], dtype=float)
                ours_seed[s] = float(np.nanmean(vals))
            ours_vals = np.array([ours_seed[s] for s in SEEDS], dtype=float)
            row["Ours_mean"] = float(np.nanmean(ours_vals))
            row["Ours_std"] = float(np.nanstd(ours_vals))

            summary_rows.append(row)

        final_df = pd.DataFrame(summary_rows)

        # 你可以按需要选择列的顺序，这里给一个典型顺序
        # 先 model / input_max_length，然后所有宏任务 mean/std，最后 Ours
        ordered_cols = ["model", "input_max_length"]
        for macro in [k for k in custom_avgs.keys() if k != "Ours"]:
            ordered_cols.append(f"{macro}_mean")
            ordered_cols.append(f"{macro}_std")
        ordered_cols += ["Ours_mean", "Ours_std"]

        final_df = final_df[ordered_cols]

        with open("result_all.csv", "w") as f:
            print(final_df.to_csv(index=False), file=f)

        print(
            "Warning, failed to get the following paths, make sure that these are correct or the printed results will not be accurate:",
            failed_paths,
        )
