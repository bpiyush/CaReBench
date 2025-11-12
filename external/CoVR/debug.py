from src.data.webvid_covr import WebVidCoVRTestDataModule, WebVidCoVRDataset


if __name__ == "__main__":
    print("Loading test loader")
    test_loader = WebVidCoVRTestDataModule(
        annotation="/scratch/shared/beegfs/piyush/datasets/WebVid-CoVR/webvid8m-covr_test-cleaned.csv",
        batch_size=8,
        vid_dirs="/datasets/WebVid/videos",
        emb_dirs="/scratch/shared/beegfs/piyush/datasets/WebVid-CoVR/blip-vid-embs-large-all",
    )
    import ipdb; ipdb.set_trace()
    for batch in test_loader:
        print(batch)
        break