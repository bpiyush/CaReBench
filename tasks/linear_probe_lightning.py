"""Linear probe parallelised with pytorch lightning."""
import os
import sys

import torch
import lightning as L
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import shared.utils as su

from tasks.linear_probe_video_embeddings import (
    DATA_CONFIG,
    read_args,
    load_model,
    load_data,
    get_linear_probe_accuracy,
)


class VideoDataset(Dataset):
    def __init__(self, df, video_processor):
        self.df = df.reset_index(drop=True)
        self.video_processor = video_processor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        video_path = row['video_path']
        video = self.video_processor(video_path)
        _id = row['id']
        return {
            'video': video,
            'id': _id,
        }


class FeatureComputer(L.LightningModule):
    def __init__(self, model_path_or_name):
        super().__init__()
        self.valid_feats = {}
        self.model_path_or_name = model_path_or_name
    
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        video = batch['video']
        _id = batch['id']
        with torch.no_grad():
            features = self.encoder.encode_vision(video).float()
            features = torch.nn.functional.normalize(features, dim=-1).cpu()
        
        # Store features for each ID in the batch
        for i in range(len(_id)):
            # Convert tensor to Python scalar if needed
            id_val = _id[i].item() if torch.is_tensor(_id[i]) else _id[i]
            self.valid_feats[id_val] = features[i]
        
        return {"loss": 0, "features": features, "ids": _id}
    
    def on_validation_epoch_end(self):
        # Return features for gathering
        return self.valid_feats
    
    def configure_model(self):
        model_path_or_name = self.model_path_or_name
        su.log.print_update(f"Loading model ({model_path_or_name}).")

        # Load model
        from models.modeling_encoders import AutoEncoder
        self.encoder = AutoEncoder.from_pretrained(
            model_path_or_name,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        su.misc.num_params(self.encoder.model)

# Define a video processor: video_path -> video_tensor
from utils.video import read_frames_decord
class VideoProcessor:
    def __init__(self, n_frames=16):
        self.n_frames = n_frames
    
    def __call__(self, video_path):
        try:
            video = read_frames_decord(video_path, self.n_frames, width=480, height=270)
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            video = torch.zeros(self.n_frames, 3, 270, 480)
        return video


if __name__ == "__main__":
    args = read_args()
    
    # Load model
    # vfc, vp = load_model(args.model_path_or_name, device_map='cpu')
    vp = VideoProcessor(n_frames=16)
    
    # Load data
    df = load_data(args.dataset, args.debug)
    ds = VideoDataset(df, vp)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(dl))
    
    # Compute features
    fc = FeatureComputer(args.model_path_or_name)
    # fc.configure_model(args.model_path_or_name)
    # feats = fc.validate(dataloader=dl)
    
    # Define trainer
    trainer = L.Trainer(
        max_epochs=1,
        devices=torch.cuda.device_count() if not args.debug else 1,
        accelerator='gpu',
        strategy='ddp',
        # precision='bf16-mixed',
    )
    # Get features from validation
    results = trainer.validate(fc, dataloaders=dl)
    
    # Handle DDP feature collection
    if trainer.world_size > 1:
        # In DDP, we need to gather features from all processes
        import torch.distributed as dist
        
        # Get features from current process
        local_feats = fc.valid_feats
        print(f"Rank {trainer.global_rank}: Local features count: {len(local_feats)}")
        
        # Gather from all processes to rank 0
        if trainer.global_rank == 0:
            gathered_feats = [None] * trainer.world_size
            dist.gather_object(local_feats, gathered_feats if trainer.global_rank == 0 else None, dst=0)
            
            # Combine all features on rank 0
            video_feat = {}
            for rank_feats in gathered_feats:
                video_feat.update(rank_feats)
            
            print(f"Rank 0: Total gathered features: {len(video_feat)}")
        else:
            dist.gather_object(local_feats, None, dst=0)
            video_feat = {}
    else:
        video_feat = fc.valid_feats
    
    # Only rank 0 computes accuracy
    if trainer.global_rank == 0 or trainer.world_size == 1:
        print("Number of features: ", len(video_feat))

        # Compute accuracy
        id_to_label = {k: v for k, v in zip(df.id, df['class'])}
        train_ids = df[df.split == 'train'].id.unique()
        valid_ids = df[df.split == 'test'].id.unique()
        train_feat = torch.stack([video_feat[k] for k in train_ids])
        valid_feat = torch.stack([video_feat[k] for k in valid_ids])
        train_labels = [id_to_label[k] for k in train_ids]
        valid_labels = [id_to_label[k] for k in valid_ids]
        valid_acc = get_linear_probe_accuracy(train_feat, train_labels, valid_feat, valid_labels)
        print(f"Valid accuracy: {valid_acc:.2f}")