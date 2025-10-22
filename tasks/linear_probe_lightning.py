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
        return self.video_processor(video_path)


class FeatureComputer(L.LightningModule):
    def __init__(self, model_path_or_name):
        super().__init__()
        self.valid_feats = []
        self.model_path_or_name = model_path_or_name
    
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            features = self.encoder.encode_vision(batch).float()
        # features = self(batch)
            features = torch.nn.functional.normalize(features, dim=-1).cpu()
        self.valid_feats.append(features)
        return {"loss": 0}
    
    def on_validation_epoch_end(self):
        valid_feats = torch.cat(self.valid_feats)
        self.valid_feats = []
        return valid_feats
    
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
    feats = trainer.validate(fc, dataloaders=dl)
    import ipdb; ipdb.set_trace()

    
