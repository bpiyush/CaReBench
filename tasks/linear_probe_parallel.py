"""Linear probe for video classification with multi-GPU support."""
import os
import sys
import pickle

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

import shared.utils as su


DATA_CONFIG = {
    "ucf101": {
        'anno_path': '/scratch/shared/beegfs/piyush/datasets/UCF101/metadata/all01.csv',
        'video_dir': '/scratch/shared/beegfs/piyush/datasets/UCF101/videos_mp4',
        'ext': 'mp4',
    }
}


def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument(
        '-d', '--dataset', type=str, required=True, choices=DATA_CONFIG.keys(),
    )
    # Model path or name
    parser.add_argument(
        '-m', '--model_path_or_name', type=str, required=True,
        default="/work/piyush/pretrained_checkpoints/CaRe-7B",
    )
    # Parallelization
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size per GPU for feature extraction'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of workers for data loading'
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Debug mode: only process the first 1000 videos'
    )
    return parser.parse_args()


class VideoDataset(Dataset):
    """Dataset for loading videos."""
    
    def __init__(self, df, data_config, n_frames=16):
        self.df = df.reset_index(drop=True)
        self.data_config = data_config
        self.n_frames = n_frames
        
        from utils.video import read_frames_decord
        self.read_frames = read_frames_decord
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        video_id = row['id']
        
        try:
            video = self.read_frames(
                video_path, self.n_frames, width=480, height=270
            )
        except Exception as e:
            if idx == 0 or idx % 100 == 0:  # Only print occasionally
                print(f"Error reading video {video_path}: {e}")
            video = torch.zeros(self.n_frames, 3, 270, 480)
        
        return {
            'video': video,
            'idx': idx,  # Use index instead of string ID
        }


def collate_fn(batch):
    """Custom collate function."""
    videos = torch.stack([item['video'] for item in batch])
    indices = torch.tensor([item['idx'] for item in batch])
    return {
        'video': videos,
        'idx': indices,
    }


def load_model(model_path_or_name='CaRe-7B', device='cuda'):
    """Load the encoder model."""
    
    su.log.print_update(f"Loading model ({model_path_or_name}).")

    # Load model
    from models.modeling_encoders import AutoEncoder
    encoder = AutoEncoder.from_pretrained(
        model_path_or_name,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    su.misc.num_params(encoder.model)
    
    # encoder.eval()
    return encoder


def compute_features_parallel(encoder, dataloader, df, accelerator):
    """Compute features in parallel across GPUs."""
    
    # Create a tensor to store all features
    num_videos = len(df)
    feature_dim = None
    all_features_dict = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(su.log.tqdm_iterator(
            dataloader, 
            desc=f'Computing features [GPU {accelerator.process_index}]',
            disable=not accelerator.is_local_main_process
        )):
            videos = batch['video']
            indices = batch['idx']
            
            # Encode videos
            vision_emb = encoder.encode_vision(videos)
            
            # Normalize features
            vision_emb = torch.nn.functional.normalize(vision_emb, dim=-1)
            
            # Store features with their indices locally
            for i, idx in enumerate(indices):
                idx_item = idx.item()
                all_features_dict[idx_item] = vision_emb[i].cpu().float()
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print(f"\nGathering features from {accelerator.num_processes} GPUs...")
    
    # Create a shared temporary directory - all processes use the same path
    import tempfile
    import torch.distributed as dist
    
    if accelerator.is_main_process:
        temp_dir = tempfile.mkdtemp()
        # Convert string to list for broadcasting
        temp_dir_list = [temp_dir]
    else:
        temp_dir_list = [None]
    
    # Broadcast the temp directory path to all processes
    dist.broadcast_object_list(temp_dir_list, src=0)
    temp_dir = temp_dir_list[0]
    
    # Save local features to temporary files
    local_file = os.path.join(temp_dir, f'features_rank_{accelerator.process_index}.pkl')
    
    with open(local_file, 'wb') as f:
        pickle.dump(all_features_dict, f)
    
    accelerator.wait_for_everyone()
    
    # Main process collects all features
    if accelerator.is_main_process:
        final_features = {}
        
        for rank in range(accelerator.num_processes):
            rank_file = os.path.join(temp_dir, f'features_rank_{rank}.pkl')
            if os.path.exists(rank_file):
                with open(rank_file, 'rb') as f:
                    rank_features = pickle.load(f)
                    final_features.update(rank_features)
                os.remove(rank_file)
            else:
                print(f"Warning: File {rank_file} not found for rank {rank}")
        
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass  # Directory not empty or other error
        
        # Convert indices back to video IDs
        video_feat = {}
        for idx, feat in final_features.items():
            video_id = df.iloc[idx]['id']
            video_feat[video_id] = feat
        
        print(f"Collected features for {len(video_feat)} videos")
        # Synchronize with other ranks before returning
        accelerator.wait_for_everyone()
        return video_feat
    else:
        # Wait for main process to finish collecting and cleaning up
        accelerator.wait_for_everyone()
        return None


def get_linear_probe_accuracy(
    X_train, Y_train, X_valid, Y_valid, verbose=True, clf="ridge"
):
    from sklearn.linear_model import RidgeClassifier

    if verbose:
        print(
            f"Fitting classifier on {X_train.shape} samples for {len(np.unique(Y_train))} classes ..."
        )
    if clf == "ridge":
        clf = RidgeClassifier()
    elif clf == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
    elif clf == "kernel_svm":
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf")
    else:
        raise ValueError(f"Unknown classifier: {clf}")
    
    clf.fit(X_train, Y_train)
    
    if verbose:
        print(f"Done fitting classifier. Evaluating ...")
    
    try:
        train_acc = np.round(
            100.0 * (clf.predict(X_train) == Y_train).float().mean().item(), 3
        )
        valid_acc = np.round(
            100.0 * (clf.predict(X_valid) == Y_valid).float().mean().item(), 3
        )
    except:
        train_acc = np.round(100.0 * (clf.predict(X_train) == Y_train).mean(), 3)
        valid_acc = np.round(100.0 * (clf.predict(X_valid) == Y_valid).mean(), 3)
    
    if verbose:
        print("." * 120)

    return valid_acc


if __name__ == "__main__":
    args = read_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Print GPU info
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"Running on {accelerator.num_processes} GPU(s)")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * accelerator.num_processes}")
        print(f"{'='*80}\n")
    
    print(f"[Rank {accelerator.process_index}] Using device: {accelerator.device}")
    accelerator.wait_for_everyone()
    
    # Load data on all processes
    if accelerator.is_main_process:
        su.log.print_update(f"Loading dataset: {args.dataset}")
    
    df = pd.read_csv(DATA_CONFIG[args.dataset]['anno_path'])
    df['video_path'] = df['id'].apply(
        lambda x: os.path.join(
            DATA_CONFIG[args.dataset]['video_dir'],
            x + f'.{DATA_CONFIG[args.dataset]["ext"]}'
        ),
    )
    
    if args.debug:
        df = df.head(1000)

    if accelerator.is_main_process:
        print("Number of rows: ", len(df))
        su.log.print_update("")
    
    # Create dataset and dataloader
    dataset = VideoDataset(df, DATA_CONFIG[args.dataset], n_frames=16)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Load model
    encoder = load_model(args.model_path_or_name, device=accelerator.device)
    
    # Prepare with accelerator
    encoder, dataloader = accelerator.prepare(encoder, dataloader)
    
    # Compute features
    video_feat = compute_features_parallel(encoder, dataloader, df, accelerator)
    
    # Only main process continues with linear probing
    if accelerator.is_main_process:
        # Compute accuracy
        id_to_label = {k: v for k, v in zip(df.id, df['class'])}
        train_ids = df[df.split == 'train'].id.unique()
        valid_ids = df[df.split == 'test'].id.unique()

        # Filter out any IDs that failed to produce features
        present_train_ids = [k for k in train_ids if k in video_feat]
        present_valid_ids = [k for k in valid_ids if k in video_feat]

        missing_train = len(train_ids) - len(present_train_ids)
        missing_valid = len(valid_ids) - len(present_valid_ids)
        if missing_train or missing_valid:
            print(f"Warning: missing features - train: {missing_train}, valid: {missing_valid}")
        
        train_feat = torch.stack([video_feat[k] for k in present_train_ids])
        valid_feat = torch.stack([video_feat[k] for k in present_valid_ids])
        train_labels = [id_to_label[k] for k in present_train_ids]
        valid_labels = [id_to_label[k] for k in present_valid_ids]
        
        valid_acc = get_linear_probe_accuracy(
            train_feat, train_labels, valid_feat, valid_labels
        )
        print(f"Valid accuracy: {valid_acc:.2f}")
    
    # Clean up process group
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()