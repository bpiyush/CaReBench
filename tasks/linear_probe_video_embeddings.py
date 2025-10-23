"""Linear probe for video classification."""
import os
import sys

import torch
import pandas as pd
import numpy as np

import shared.utils as su


DATA_CONFIG = {
    "ucf101": {
        'anno_path': '/scratch/shared/beegfs/piyush/datasets/UCF101/metadata/all01.csv',
        'video_dir': '/scratch/shared/beegfs/piyush/datasets/UCF101/videos_mp4',
        'ext': 'mp4',
    },
    "hmdb51": {
        'anno_path': '/scratch/shared/beegfs/piyush/datasets/HMDB51/metadata/all.csv',
        'video_dir': '/scratch/shared/beegfs/piyush/datasets/HMDB51/videos',
        'ext': 'avi',
    },
    "ssv2": {
        "anno_path": "/scratch/shared/beegfs/piyush/datasets/SSv2/labels/all.csv",
        "video_dir": "/scratch/shared/beegfs/piyush/datasets/SSv2/20bn-something-something-v2",
        "ext": "webm",
    },
    "k400": {
        "anno_path": "/scratch/shared/beegfs/piyush/datasets/Kinetics400/metadata/all.csv",
        "video_dir": "/datasets/KineticsClean/",
        "ext": "mp4",
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
    parser.add_argument(
        '-debug', '--debug', action='store_true', default=False,
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
    )
    parser.add_argument(
        "--gpus", type=int, default=torch.cuda.device_count(),
    )
    return parser.parse_args()


def load_model(model_path_or_name='CaRe-7B', device_map='auto'):

    su.log.print_update(f"Loading model ({model_path_or_name}).")

    # Load model
    from models.modeling_encoders import AutoEncoder
    encoder = AutoEncoder.from_pretrained(
        model_path_or_name,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
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

def compute_features(vp, vfc, df):
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


def load_data(dataset, debug=False):
    su.log.print_update(f"Loading dataset: {dataset}")
    df = pd.read_csv(DATA_CONFIG[dataset]['anno_path'])
    df['video_path'] = df['id'].apply(
        lambda x: os.path.join(
            DATA_CONFIG[dataset]['video_dir'],
            str(x) + f'.{DATA_CONFIG[dataset]["ext"]}'
        ),
    )
    df = df[df.split.isin(['train', 'test'])]
    if debug:
        indices_train = np.random.choice(np.where(df.split == 'train')[0], 200)
        indices_test = np.random.choice(np.where(df.split == 'test')[0], 200)
        df = df.iloc[np.concatenate([indices_train, indices_test])]
    print("Number of rows: ", len(df))
    su.log.print_update("")
    return df


if __name__ == "__main__":
    args = read_args()
    
    # Load data
    df = load_data(args.dataset, args.debug)

    # Load model
    vfc, vp = load_model(args.model_path_or_name)
    
    # Compute features
    video_feat = compute_features(vp, vfc, df)
    
    
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
