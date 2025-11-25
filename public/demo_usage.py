import torch
from termcolor import colored
from modeling_tara import TARA, read_frames_decord, read_images_decord


def main(model_path: str = "."):
    print(colored("="*60, 'yellow'))
    print(colored("TARA Model Demo", 'yellow', attrs=['bold']))
    print(colored("="*60, 'yellow'))
    
    # Load model from current directory
    print(colored("\n[1/3] Loading model...", 'cyan'))
    model = TARA.from_pretrained(
        model_path,  # Load from current directory
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    
    n_params = sum(p.numel() for p in model.model.parameters())
    print(colored(f"✓ Model loaded successfully!", 'green'))
    print(f"Number of parameters: {round(n_params/1e9, 3)}B")
    
    # Encode a sample video
    print(colored("\n[2/3] Testing video encoding...", 'cyan'))
    video_path = "./assets/folding_paper.mp4"
    
    try:
        video_tensor = read_frames_decord(video_path, num_frames=16)
        video_tensor = video_tensor.unsqueeze(0)
        video_tensor = video_tensor.to(model.model.device)
        
        with torch.no_grad():
            video_emb = model.encode_vision(video_tensor).cpu().squeeze(0).float()
        
        print(colored("✓ Video encoded successfully!", 'green'))
        print(f"Video shape: {video_tensor.shape}")  # torch.Size([1, 16, 3, 240, 426])
        print(f"Video embedding shape: {video_emb.shape}")  # torch.Size([4096])
    except FileNotFoundError:
        print(colored(f"⚠ Video file not found: {video_path}", 'red'))
        print(colored("  Please add a video file or update the path in demo_usage.py", 'yellow'))
        video_emb = None
    
    # Encode sample texts
    print(colored("\n[3/3] Testing text encoding...", 'cyan'))
    text = ['someone is folding a paper', 'cutting a paper', 'someone is unfolding a paper']
    # NOTE: It can also take a single string
    
    with torch.no_grad():
        text_emb = model.encode_text(text).cpu().float()
    
    print(colored("✓ Text encoded successfully!", 'green'))
    print(f"Text: {text}")
    print(f"Text embedding shape: {text_emb.shape}")  # torch.Size([3, 4096])
    
    # Compute similarities if video was encoded
    if video_emb is not None:
        print(colored("\n[Bonus] Computing video-text similarities...", 'cyan'))
        similarities = torch.cosine_similarity(
            video_emb.unsqueeze(0).unsqueeze(0),  # [1, 1, 4096]
            text_emb.unsqueeze(0),                # [1, 3, 4096]
            dim=-1
        )
        print(colored("✓ Similarities computed!", 'green'))
        for i, txt in enumerate(text):
            print(f"  '{txt}': {similarities[0, i].item():.4f}")
    print("-" * 100)
    
    
    # Negation example: a negation in text query should result
    # in retrieval of images without the neg. object in the query
    image_paths = [
        './assets/cat.png',
        './assets/dog+cat.png',
    ]
    image_tensors = read_images_decord(image_paths)
    with torch.no_grad():
        image_embs = model.encode_vision(image_tensors.to(model.model.device)).cpu().float()
        image_embs = torch.nn.functional.normalize(image_embs, dim=-1)
    print(f"Image embedding shape: {image_embs.shape}")
    
    texts = ['an image of a cat but there is no dog in it']
    with torch.no_grad():
        text_embs = model.encode_text(texts).cpu().float()
        text_embs = torch.nn.functional.normalize(text_embs, dim=-1)
    print("Text query: ", texts)
    sim = text_embs @ image_embs.t()
    print(f"Text-Image similarity: {sim}")
    print("-" * 100)
    
    texts = ['an image of a cat and a dog together']
    with torch.no_grad():
        text_embs = model.encode_text(texts).cpu().float()
        text_embs = torch.nn.functional.normalize(text_embs, dim=-1)
    print("Text query: ", texts)
    sim = text_embs @ image_embs.t()
    print(f"Text-Image similarity: {sim}")
    print("-" * 100)
    import ipdb; ipdb.set_trace()
    
    
    print(colored("\n" + "="*60, 'yellow'))
    print(colored("Demo completed successfully! 🎉", 'green', attrs=['bold']))
    print(colored("="*60, 'yellow'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=".")
    args = parser.parse_args()

    main(args.model_path)