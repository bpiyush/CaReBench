import decord
from decord import VideoReader
import decord
import random
import numpy as np

decord.bridge.set_bridge("torch")


def read_frames_decord(
        video_path, num_frames, sample='middle', fix_start=None, 
        max_num_frames=-1, trimmed30=False, height=-1, width=-1
    ):
    decord.bridge.set_bridge('torch')

    # num_threads = 1 if video_path.endswith('.webm') else 0 # make ssv2 happy
    num_threads = 1
    video_reader = VideoReader(video_path, num_threads=num_threads, height=height, width=width)
    vlen = len(video_reader)
 
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # only use top 30 seconds
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )

    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames.asnumpy())
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


import torch
import torch.nn.functional as F


def make_square_video_tensor(video: torch.Tensor, min_side: int = 336) -> torch.Tensor:
    """
    Process video tensor by resizing and center-padding.
    
    Args:
        video: Input tensor of shape [T, C, H', W']
        min_side: Target size for the minimum side (default: 336)
    
    Returns:
        Output tensor of shape [T, C, H, W] where H == W == max(resized_h, resized_w)
    """
    T, C, H, W = video.shape
    
    # Step 1: Find the minimum side
    current_min_side = min(H, W)
    
    # Step 2: Calculate scale factor and new dimensions
    scale = min_side / current_min_side
    new_h = int(H * scale)
    new_w = int(W * scale)
    
    # Resize the video tensor
    # F.interpolate expects input of shape [N, C, H, W], so it works with [T, C, H, W]
    video_resized = F.interpolate(
        video, 
        size=(new_h, new_w), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Step 3: Center-pad to make it square
    max_side = max(new_h, new_w)
    
    # Calculate padding: (left, right, top, bottom)
    pad_h = max_side - new_h
    pad_w = max_side - new_w
    
    # Distribute padding evenly on both sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding (padding order in F.pad is: left, right, top, bottom)
    video_padded = F.pad(
        video_resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=0
    )
    
    return video_padded


def pad_to_square_then_resize(video: torch.Tensor, resize_to: int) -> torch.Tensor:
    """
    Process video tensor by first padding to square, then resizing.
    
    Args:
        video: Input tensor of shape [T, C, H', W']
        resize_to: Final size to resize both dimensions to (creates square output)
    
    Returns:
        Output tensor of shape [T, C, resize_to, resize_to]
    """
    T, C, H, W = video.shape
    
    # Step 1: Pad to square by padding the minimum side to match the maximum side
    max_side = max(H, W)
    
    pad_h = max_side - H
    pad_w = max_side - W
    
    # Distribute padding evenly on both sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding (padding order in F.pad is: left, right, top, bottom)
    video_padded = F.pad(
        video,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=0
    )
    
    # Step 2: Resize to target size
    video_resized = F.interpolate(
        video_padded,
        size=(resize_to, resize_to),
        mode='bilinear',
        align_corners=False
    )
    
    return video_resized


# Example usage
if __name__ == "__main__":
    # Example 1: Height < Width
    # Input: [T=10, C=3, H=240, W=320]
    video_input = torch.randn(10, 3, 240, 320)
    video_output = pad_to_square_then_resize(video_input, resize_to=448)
    
    print(f"Input shape: {video_input.shape}")
    print(f"After padding to square: [10, 3, 320, 320]")
    print(f"After resizing to 448: {video_output.shape}")
    # Expected: torch.Size([10, 3, 448, 448])
    
    # Example 2: Width < Height
    # Input: [T=5, C=3, H=480, W=320]
    video_input2 = torch.randn(5, 3, 480, 320)
    video_output2 = pad_to_square_then_resize(video_input2, resize_to=512)
    
    print(f"\nInput shape: {video_input2.shape}")
    print(f"After padding to square: [5, 3, 480, 480]")
    print(f"After resizing to 512: {video_output2.shape}")
    # Expected: torch.Size([5, 3, 512, 512]


# Example usage
if __name__ == "__main__":
    # Test with the example from the description
    # Input: [T=10, C=3, H=240, W=320]
    video_input = torch.randn(10, 3, 240, 320)
    video_output = make_square_video_tensor(video_input, min_side=336)
    
    print(f"Input shape: {video_input.shape}")
    print(f"Output shape: {video_output.shape}")
    # Expected: torch.Size([10, 3, 448, 448])
    
    # Test with width as minimum side
    video_input2 = torch.randn(5, 3, 480, 320)
    video_output2 = make_square_video_tensor(video_input2, min_side=336)
    
    print(f"\nInput shape: {video_input2.shape}")
    print(f"Output shape: {video_output2.shape}")
    # Expected: torch.Size([5, 3, 504, 504])