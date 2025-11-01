import os
import sys
import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import einops

import shared.utils as su

os.environ['TOKENIZERS_PARALLELISM'] = "False"
plt.rcParams["font.family"] = "serif"


DATA_ROOT = "/scratch/shared/beegfs/piyush/datasets"
VIDEO_DIR = {
    "ssv2": f"{DATA_ROOT}/SSv2/20bn-something-something-v2",
    "epic": f"{DATA_ROOT}/EPIC-Kitchens-100/cut_clips",
    "charades": f"{DATA_ROOT}/Charades/Charades_v1_480_cut_clips"
}
EXT = {
    'ssv2': 'webm',
    'epic': 'MP4',
    'charades': 'mp4',
}
REPO_PATH = "/users/piyush/projects/TimeBound.v1/"
SPLIT_DIR = f"{REPO_PATH}/adapt4change/chirality_in_action_splits"


def load_data(dataset='ssv2', split='validation'):
    assert split in ['train', 'validation']
    
    # Pick CSV path
    csv_path = f"{SPLIT_DIR}/cia-{dataset}-{split}.csv"
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)

    # Add text ID
    df['text_id'] = df[['chiral_triplet_id', 'chiral_label']].apply(
        lambda x: f"{x[0]}_{x[1]}", axis=1,
    )
    video_dir = VIDEO_DIR[dataset]
    ext = EXT[dataset]

    df['video_path'] = df['id'].apply(lambda x: f"{video_dir}/{x}.{ext}")
    df = df[df.video_path.apply(os.path.exists)]
    print("Number of rows: ", len(df))
    print("Sample row: ")
    print(json.dumps(df.iloc[0].to_dict(), indent=4))
    
    return df


def load_model(model_path_or_name='CaRe-7B'):

    su.log.print_update(f"Loading model ({model_path_or_name}).")

    # Load model
    from models.modeling_encoders import AutoEncoder
    encoder = AutoEncoder.from_pretrained(
        model_path_or_name,
        device_map='auto',
        attn_implementation="flash_attention_2",
    )
    su.misc.num_params(encoder.model)

    # Define a feature computer: video_tensor -> video_feature
    class VideoFeatureComputer:
        def __init__(self, encoder):
            self.encoder = encoder
        
        def __call__(self, video_tensor):
            with torch.no_grad():
                vision_emb = encoder.encode_vision(
                    video_tensor.unsqueeze(0),
                ).cpu().squeeze(0).float()
            return vision_emb
    vfc = VideoFeatureComputer(encoder)


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
    vp = VideoProcessor(n_frames=16)
    return vfc, vp

def compute_features(df):
    video_paths = df.video_path.unique()
    video_ids = df.id.unique()
    video_feat = {}
    j = 0
    for video_path in su.log.tqdm_iterator(video_paths, desc='Computing video features'):
        video_tensor = vp(video_path).cuda()
        zv = vfc(video_tensor)
        zv = torch.nn.functional.normalize(zv, dim=-1)
        video_feat[video_ids[j]] = zv
        j += 1
    return video_feat


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

def compute_chiral_accuracy(
    X_train, X_valid, df_train, df_valid, match_col="chiral_triplet_id", verbose=True, return_all=False,
):
    accs = []
    data = {"triplet_id": [], "acc": []}
    triplet_ids = df_train[match_col].unique()
    if verbose:
        iterator = su.log.tqdm_iterator(triplet_ids, desc="Evaluating chiral accuracy")
    else:
        iterator = triplet_ids
    for tid in iterator:
        idx_train = np.where(df_train[match_col] == tid)[0]
        idx_valid = np.where(df_valid[match_col] == tid)[0]
        x_train = X_train[idx_train]
        y_train = df_train.iloc[idx_train]["chiral_label"].values
        x_valid = X_valid[idx_valid]
        y_valid = df_valid.iloc[idx_valid]["chiral_label"].values
        acc = get_linear_probe_accuracy(
            x_train, y_train, x_valid, y_valid, verbose=False,
        )
        accs.append(acc)
        data["triplet_id"].append(tid)
        data["acc"].append(acc)
    avg = np.mean(accs)
    if return_all:
        return data
    return avg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ssv2')
    parser.add_argument('--model_path', type=str, default='/work/piyush/experiments/CaRe/special_milestones/care-stage2-nli90k-ego4d-10k')
    # parser.add_argument('--index', type=int, default=16607)
    args = parser.parse_args()
    
    dataset = args.dataset
    df_train = load_data(dataset=dataset, split='train')
    df_valid = load_data(dataset=dataset, split='validation')
    
    model_path_or_name = args.model_path
    vfc, vp = load_model(model_path_or_name)

    # i = 16607
    # row = df_train.iloc[i]
    # video_path = row.video_path
    # video_tensor = vp(video_path)
    # video_feat = vfc(video_tensor)
    # print(video_feat.shape)
    
    video_feat_train = compute_features(df_train)
    video_feat_valid = compute_features(df_valid)
    
    X_train = torch.stack([video_feat_train[k] for k in df_train.id])
    X_valid = torch.stack([video_feat_valid[k] for k in df_valid.id])
    accu = compute_chiral_accuracy(X_train, X_valid, df_train, df_valid)
    print(f"Chiral accuracy: {accu:.2f}%")
