import os
import argparse
import numpy as np

import shared.utils as su


def format_result_item(item, index):
    video = item.get('video', 'N/A')
    question = item.get('question', 'N/A')
    generated_answer = item.get('generated_answer', 'N/A')
    true_answer = item.get('true_answer', 'N/A')
    options = item.get('options', [])

    # Truncate long strings for neat printing
    def truncate(text, max_len=200):
        s = str(text)
        return s if len(s) <= max_len else s[: max_len - 3] + '...'

    lines = [
        f"[{index}] video: {video}",
        f"     question: {truncate(question)}",
        f"     generated_answer: {truncate(generated_answer)}",
        f"     true_answer: {truncate(true_answer)}",
    ]

    if isinstance(options, (list, tuple)) and options:
        opt_lines = []
        for j, opt in enumerate(options):
            opt_lines.append(f"       - {j}: {truncate(opt, 180)}")
        lines.append("     options:")
        lines.extend(opt_lines)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check and print first 10 results from .npy file")
    default_path = os.path.join(su.log.repo_path, 'results', 'nextqa_mc.npy')
    parser.add_argument('--path', type=str, default=default_path, help='Path to the .npy results file')
    parser.add_argument('--num', type=int, default=10, help='Number of results to display')
    args = parser.parse_args()

    path = args.path
    assert os.path.exists(path), f"Results file not found: {path}"

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.tolist()

    assert isinstance(data, list), "Expected a list of result dicts in the .npy file"

    total = len(data)
    print(f"Loaded {total} results from: {path}")
    print("=" * 80)

    to_show = min(args.num, total)
    for i in range(to_show):
        item = data[i]
        if not isinstance(item, dict):
            print(f"[{i}] Non-dict entry: {type(item)}")
        else:
            print(format_result_item(item, i))
        print("-" * 80)


if __name__ == '__main__':
    main()


