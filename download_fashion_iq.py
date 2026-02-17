"""
Download Fashion-IQ images from Amazon URLs.

Reads asin2url.{dress,shirt,toptee}.txt files from the metadata directory,
downloads each image in parallel, and saves as {ASIN}.jpg.
Skips already-downloaded images for resume capability.
"""

import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

METADATA_DIR = "/scratch/shared/beegfs/piyush/datasets/CoIR/fashion-iq-metadata/image_url"
OUTPUT_DIR = "/scratch/shared/beegfs/piyush/datasets/CoIR/fashion-iq-images"
CATEGORIES = ["dress", "shirt", "toptee"]
NUM_WORKERS = 4
TIMEOUT = 15  # seconds per request


def load_asin_urls():
    """Load all ASIN -> URL mappings from the metadata files."""
    asin2url = {}
    for cat in CATEGORIES:
        filepath = os.path.join(METADATA_DIR, f"asin2url.{cat}.txt")
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    asin = parts[0].strip()
                    url = parts[1].strip()
                    asin2url[asin] = url
    return asin2url


def download_image(asin, url, output_dir):
    """Download a single image. Returns (asin, success, error_msg)."""
    output_path = os.path.join(output_dir, f"{asin}.jpg")
    if os.path.exists(output_path):
        return asin, True, "skipped (exists)"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and len(resp.content) < 1000:
            return asin, False, f"not an image (Content-Type: {content_type}, size: {len(resp.content)})"
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return asin, True, "ok"
    except Exception as e:
        return asin, False, str(e)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ASIN -> URL mappings...")
    asin2url = load_asin_urls()
    print(f"Loaded {len(asin2url)} unique ASINs across {len(CATEGORIES)} categories.")

    # Check how many already exist
    already_done = sum(
        1 for asin in asin2url if os.path.exists(os.path.join(OUTPUT_DIR, f"{asin}.jpg"))
    )
    remaining = len(asin2url) - already_done
    print(f"Already downloaded: {already_done}, Remaining: {remaining}")

    if remaining == 0:
        print("All images already downloaded!")
        return

    failed = []
    success_count = 0
    skip_count = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(download_image, asin, url, OUTPUT_DIR): asin
            for asin, url in asin2url.items()
        }
        with tqdm(total=len(futures), desc="Downloading images") as pbar:
            for future in as_completed(futures):
                asin, ok, msg = future.result()
                if ok:
                    if msg == "skipped (exists)":
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    failed.append((asin, msg))
                pbar.update(1)

    print(f"\nDone! New downloads: {success_count}, Skipped: {skip_count}, Failed: {len(failed)}")

    if failed:
        fail_path = os.path.join(OUTPUT_DIR, "failed_downloads.txt")
        with open(fail_path, "w") as f:
            for asin, msg in sorted(failed):
                f.write(f"{asin}\t{msg}\n")
        print(f"Failed ASINs logged to: {fail_path}")


if __name__ == "__main__":
    main()
