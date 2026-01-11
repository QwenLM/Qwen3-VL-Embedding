import os
import sys
import time
import yaml
import torch
import random
import datetime
import pickle
import json
import numpy as np
import torch.distributed as dist

from datetime import timedelta
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node
from .arguments import ModelArguments, DataArguments, EvalArguments
from .utils.basic_utils import print_rank, print_master
from .utils.eval_utils.metrics import RankingMetrics
from .models import MMEBEmbeddingModel
from .data.datasets.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from .data.collator import MultimodalEvalDataCollator

def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size


@torch.no_grad()
def encode_embeddings(
    model: MMEBEmbeddingModel,
    loader: DataLoader,
    encode_side: str,  # 'qry' or 'cand'
    full_dataset_len: int,
    description: str = "Encoding"
) -> tuple[np.ndarray, list]:
    """
    Generate embeddings using MMEBEmbeddingModel's encode_input method.
    Supports DDP distributed gathering.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_embeds = []
    local_gt_infos = []

    model.eval()
    
    # Show tqdm progress bar only on main process
    progress_bar = tqdm(loader, desc=f"{description} (rank {rank})", disable=rank > 0)

    for batch_inputs, dataset_info in progress_bar:
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            reps = model.encode_input(batch_inputs)  # Returns [Batch, Dim]
            reps = reps.detach()

        local_embeds.append(reps)
        
        # Process metadata
        if encode_side == "qry":
            local_gt_infos.extend(dataset_info)
        else:
            local_gt_infos.extend([info.get("cand_name", "") for info in dataset_info])

    if not local_embeds:
        return np.array([]), []
    
    local_embeds_tensor = torch.cat(local_embeds, dim=0).contiguous()

    # DDP synchronization logic
    if dist.is_initialized():
        # Prepare global container
        embed_dim = local_embeds_tensor.shape[-1]
        all_reps_tensor = torch.zeros(full_dataset_len, embed_dim, 
                                     dtype=local_embeds_tensor.dtype, 
                                     device=local_embeds_tensor.device)
        
        # Gather tensors
        gathered_embeds = [torch.zeros_like(local_embeds_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_embeds, local_embeds_tensor)
        
        # Gather metadata
        gathered_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_infos, local_gt_infos)
        
        if rank == 0:
            final_embeddings = torch.cat(gathered_embeds, dim=0).cpu().float().numpy()
            final_infos = [info for rank_list in gathered_infos for info in rank_list]
            
            # Truncate potential DDP padding to match original dataset size
            return final_embeddings[:full_dataset_len], final_infos[:full_dataset_len]
        else:
            return None, None
    else:
        return local_embeds_tensor.cpu().float().numpy(), local_gt_infos

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print_master("=== Distributed Setup Initialized ===")
    print_master(f"Master Info -> ADDR: {os.environ.get('MASTER_ADDR')}, PORT: {os.environ.get('MASTER_PORT')}")
    print_master(f"Global World Size: {world_size}")
    print_rank(f"Process Identity -> Rank: {rank}, Local Rank: {local_rank} on {torch.cuda.get_device_name()}")

    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    eval_args: EvalArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    # DDP-safe model loading
    # Step 1: Only rank 0 downloads the model
    if rank == 0:
        model = MMEBEmbeddingModel.load(
            model_name_or_path=model_args.model_name_or_path,
            normalize=model_args.normalize,
            instruction=model_args.instruction,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
        )
        print_master(f"[rank=0] Loading the model from: {model_args.model_name_or_path}...")

    # Step 2: All processes wait until rank 0 finishes downloading
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Step 3: Non-master processes load from local cache
    if rank != 0:
        print_rank(f"Loading the model from cache...")
        time.sleep(random.randint(2 * rank, 3 * rank))
        model = MMEBEmbeddingModel.load(
            model_name_or_path=model_args.model_name_or_path,
            normalize=model_args.normalize,
            instruction=model_args.instruction,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
        )
    
    model.eval()
    model = model.to(eval_args.device, dtype=torch.bfloat16)
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    # Main evaluation loop
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):
        # 0. load dataset
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)

        if do_query or do_cand:
            if data_args.data_basedir is not None:
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if task_config.get(key):
                        task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

            try:
                full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(
                    model_args=model_args, data_args=data_args, **task_config
                )
                full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
                eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
                
                # Pad datasets to be divisible by world_size before splitting
                if dist.is_initialized():
                    padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                    padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)
                    eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=rank, world_size=world_size)
                    eval_cand_dataset = split_dataset_by_node(padded_cand_dataset, rank=rank, world_size=world_size)
                else:
                    padded_qry_dataset, padded_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
            except Exception as e:
                print_master(f"Failed to load dataset {dataset_name}, skipping {dataset_name}")
                import traceback
                traceback.print_exc()
                print_master(e)
                raise e

        # 1. Compute query embeddings
        if do_query:
            print_master("Encoding queries...")
            eval_qry_collator = MultimodalEvalDataCollator(encode_side="qry")
            eval_qry_loader = DataLoader(
                eval_qry_dataset, 
                batch_size=eval_args.per_device_eval_batch_size, 
                collate_fn=eval_qry_collator, 
                num_workers=eval_args.dataloader_num_workers,
                pin_memory=True,
                shuffle=False  # Must disable shuffle for encoding
            )
            query_embeds, gt_infos = encode_embeddings(
                model=model,
                loader=eval_qry_loader,
                encode_side="qry",
                full_dataset_len=len(full_eval_qry_dataset),
                description=f"Queries: {dataset_name}"
            )
            if rank == 0:
                query_embeds = query_embeds[:len(full_eval_qry_dataset)]
                gt_infos = gt_infos[:len(full_eval_qry_dataset)]
                os.makedirs(os.path.dirname(query_embed_path), exist_ok=True)

                # Save embeddings
                with open(query_embed_path, 'wb') as f:
                    pickle.dump(query_embeds, f)
                    
                # Save dataset info in JSONL format
                with open(dataset_info_path, 'w', encoding='utf-8') as f:
                    for info in gt_infos:
                        f.write(json.dumps(info, ensure_ascii=False) + '\n')
                        
                print_master(f"Successfully saved {len(query_embeds)} query embeddings to {query_embed_path}")

            if dist.is_initialized():
                dist.barrier()

        # 2. Compute candidate embeddings
        if do_cand:
            print_master("Encoding candidates...")
            eval_cand_collator = MultimodalEvalDataCollator(encode_side="cand")
            eval_cand_loader = DataLoader(
                eval_cand_dataset, 
                batch_size=eval_args.per_device_eval_batch_size, 
                collate_fn=eval_cand_collator, 
                num_workers=eval_args.dataloader_num_workers,
                pin_memory=True,
                shuffle=False
            )
            cand_embeds, all_cand_ids = encode_embeddings(
                model=model,
                loader=eval_cand_loader,
                encode_side="cand",
                full_dataset_len=len(full_eval_cand_dataset),
                description=f"Candidates: {dataset_name}"
            )
            if rank == 0:
                cand_embeds = cand_embeds[:len(full_eval_cand_dataset)]
                all_cand_ids = all_cand_ids[:len(full_eval_cand_dataset)]

                os.makedirs(os.path.dirname(cand_embed_path), exist_ok=True)

                # Map embeddings to dictionary: {cand_id: embedding_vector}
                # Enables fast lookup by ID during retrieval evaluation
                cand_embed_dict = {
                    cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)
                }
                with open(cand_embed_path, 'wb') as f:
                    pickle.dump(cand_embed_dict, f)
                print_master(f"Successfully saved {len(cand_embed_dict)} unique candidate embeddings to {cand_embed_path}")

            if dist.is_initialized():
                dist.barrier()
        
        # 3. Compute scores (rank 0 only)
        if rank == 0:
            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
            pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")
            
            # Skip computation only if both files exist and are valid
            need_compute = True
            if os.path.exists(score_path) and os.path.exists(pred_path):
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    if "num_pred" in score_dict:
                        print_master(f"Results already exist for {dataset_name}. Skipping computation.")
                        formatted = {k: f"{v:.4f}" for k, v in score_dict.items() if isinstance(v, (int, float))}
                        print_master(f"Scores: {formatted}")
                        need_compute = False
                except Exception as e:
                    print_master(f"Cache for {dataset_name} is corrupted ({e}), re-computing...")

            if need_compute:
                # Load persisted embeddings and metadata
                with open(query_embed_path, 'rb') as f:
                    qry_embeds = pickle.load(f)  # np.ndarray [Nq, D]
                with open(cand_embed_path, 'rb') as f:
                    cand_embed_dict = pickle.load(f)  # Dict {id: [D]}
                # Explicitly specify UTF-8 encoding to handle non-ASCII characters in dataset metadata
                gt_infos = [json.loads(l) for l in open(dataset_info_path, encoding='utf-8')]
                
                device = model.device
                pred_dicts = []
                
                # Convert to tensors and compute on GPU for acceleration
                qry_tensor = torch.from_numpy(qry_embeds).to(device)
                
                rank_against_all_candidates = task_config.get("eval_type", "global") == "global"
                if rank_against_all_candidates:
                    # Global retrieval
                    cand_keys = list(cand_embed_dict.keys())
                    cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])
                    cand_tensor = torch.from_numpy(cand_embeds).to(device)
                    
                    with torch.no_grad():
                        # Compute similarity matrix [Nq, Nc] using model's matmul
                        scores = model.compute_similarity(qry_tensor, cand_tensor)
                        # Get ranked indices (descending order)
                        _, ranked_indices = torch.sort(scores, dim=1, descending=True)
                        ranked_indices = ranked_indices.cpu().numpy()
                    
                    del cand_tensor
                    torch.cuda.empty_cache()

                    for qid, (ranked_idx, gt_info) in tqdm(enumerate(zip(ranked_indices, gt_infos)), 
                                                        total=len(gt_infos), desc=f"Global Ranking: {dataset_name}"):
                        rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                        rel_scores = gt_info.get("rel_scores", None)
                        
                        pred_dicts.append({
                            "prediction": [cand_keys[i] for i in ranked_idx],  # Save complete ranking
                            "label": rel_docids,
                            "rel_scores": rel_scores,
                        })
                else:
                    # Local ranking (in-batch or per-query set)
                    for qid, (qry_vec, gt_info) in tqdm(enumerate(zip(qry_tensor, gt_infos)), 
                                                        total=len(gt_infos), desc=f"Local Ranking: {dataset_name}"):
                        cand_names = gt_info["cand_names"]
                        cand_embeds = np.stack([cand_embed_dict[name] for name in cand_names])
                        cand_tensor = torch.from_numpy(cand_embeds).to(device)
                        
                        with torch.no_grad():
                            # Compute single-row similarity [1, Nc]
                            sim_scores = model.compute_similarity(qry_vec.unsqueeze(0), cand_tensor).squeeze(0)
                            _, ranked_idx = torch.sort(sim_scores, descending=True)
                            ranked_idx = ranked_idx.cpu().numpy()

                        rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                        rel_scores = gt_info.get("rel_scores", None)

                        pred_dicts.append({
                            "prediction": [cand_names[i] for i in ranked_idx],
                            "label": rel_docids,
                            "rel_scores": rel_scores,
                        })
                    
                    torch.cuda.empty_cache()

                # Compute metrics
                metrics_to_report = task_config.get("metrics", ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"])
                metrics = RankingMetrics(metrics_to_report)
                score_dict = metrics.evaluate(pred_dicts)
                
                score_dict["num_pred"] = len(pred_dicts)
                score_dict["num_data"] = len(gt_infos)
                
                # Persist results
                with open(score_path, "w") as f:
                    json.dump(score_dict, f, indent=4)
                
                # Save predictions in JSONL format (complete candidate ranking)
                with open(pred_path, "w", encoding='utf-8') as f:
                    for pred in pred_dicts:
                        f.write(json.dumps(pred, ensure_ascii=False) + '\n')
                
                formatted = {k: f"{v:.4f}" for k, v in score_dict.items() if isinstance(v, (int, float))}
                print_master(f"Final Score for {dataset_name}: {formatted}")

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
