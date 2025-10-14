# Python function to replace Ego4D anonymized labels like "#C C" or "#X X" with realistic nouns/pronouns.
# The function keeps consistent replacements per label within a single caption.
# It handles multiple labels, punctuation, and capitalizes replacements when they start a sentence.
# Example usage and tests are shown below.

import re
import random

def replace_ego4d_labels(caption: str) -> str:
    """
    Replace anonymized Ego4D labels of the form "#X X" (e.g., "#C C") with realistic nouns/pronouns.
    The same label (e.g., "#C") will map to the same replacement within a single caption to preserve coherence.
    
    Args:
        caption: original caption string containing labels like "#C C"
    Returns:
        cleaned caption with labels replaced by natural-sounding noun/pronoun phrases
    """
    # Diverse pool of replacement phrases (no proper nouns)
    replacements_pool = [
        "a person", "someone", "the person", "a man", "a woman", "the man", "the woman",
        "they", "he", "she", "the fellow", "the lady", "an individual", "a passerby",
        "a worker", "a student", "a shopper", "a driver", "a cyclist", "a child",
        "a teenager", "an elderly man", "an elderly woman", "an adult", "a customer",
        "a patron", "a friend", "a neighbor", "a parent", "a sibling", "a bystander",
        "a pedestrian", "the person in the frame", "a person nearby", "a cook", "a chef",
        "a nurse", "a doctor", "a teacher", "a passerby", "a youngster", "a kid"
    ]
    
    # Shuffle pool for randomness without repetition preference
    random.shuffle(replacements_pool)
    pool_iter = iter(replacements_pool)
    
    # Map to store consistent replacement per label for this caption
    label_map = {}
    
    # Pattern to match labels of the form "#X X" (where the token after the space repeats the label without '#')
    # Examples matched: "#C C", "#X X", "#ID ID" (works for multiplename tokens too)
    pattern = re.compile(r'#([A-Za-z0-9]+)\s+\1\b')
    
    def choose_replacement(label: str):
        # If we've already chosen for this label, reuse it
        if label in label_map:
            return label_map[label]
        # Otherwise pick next unused from pool; if exhausted, pick random from pool (allow repeats)
        try:
            choice = next(pool_iter)
        except StopIteration:
            choice = random.choice(replacements_pool)
        label_map[label] = choice
        return choice
    
    # Function used for regex substitution; receives a match object
    def repl(match):
        label = match.group(1)  # e.g., 'C' for "#C C"
        choice = choose_replacement(label)
        
        # Determine whether the replacement needs capitalization.
        # If the match occurs at start of string or right after a sentence-ending punctuation, capitalize.
        start_index = match.start()
        needs_cap = False
        if start_index == 0:
            needs_cap = True
        else:
            # Look at the character preceding the match (ignoring whitespace)
            prev_chunk = caption[:start_index]
            # Find last non-space character if any
            m = re.search(r'(\S)\s*$', prev_chunk)
            if m and m.group(1) in '.!?':
                needs_cap = True
        
        if needs_cap:
            # Capitalize only the first character of the chosen phrase (keep rest as-is)
            choice = choice[0].upper() + choice[1:]
        return choice
    
    # Replace all occurrences; use a lambda that refers back to original caption for punctuation detection
    # Because repl needs access to the original caption to detect context, we'll perform iterative substitution
    # to update caption as we go and preserve correct index positions for subsequent matches.
    out = caption
    offset = 0
    # We'll find matches in the current out string repeatedly
    while True:
        m = pattern.search(out)
        if not m:
            break
        # Compute replacement using the slice from the original 'caption' up to current match start to check punctuation context
        # Determine absolute start in 'out' (not necessary here since we use 'out' directly)
        replacement = repl(m)
        # Do the actual replacement in 'out'
        out = out[:m.start()] + replacement + out[m.end():]
    
    return out


# --- Example tests ---
examples = [
    "#C C picks a cloth from the drying rack.",
    "#X X talks to #Y Y.",
    "In the clip #A A opens the door and #A A steps out.",
    "#B B: hands #C C a cup.",
    "#D D smiles. #D D waves back.",
    "At the start, #E E sits down; later #E E stands up.",
    "Multiple people: #X X and #Y Y and #Z Z move around."
]

for ex in examples:
    print("Original:", ex)
    print("Cleaned: ", replace_ego4d_labels(ex))
    print()


import re
import random

def replace_ego4d_labels_batch(captions):
    """
    Process a list of Ego4D captions, replacing anonymized tokens (e.g. "#C C")
    with realistic nouns/pronouns. The same label is replaced consistently
    across all captions in the batch.

    Args:
        captions (list[str]): list of caption strings

    Returns:
        list[str]: processed captions
    """
    replacements_pool = [
        "a person", "someone", "the person", "a man", "a woman", "the man", "the woman",
        "they", "he", "she", "the fellow", "the lady", "an individual", "a passerby",
        "a worker", "a student", "a shopper", "a driver", "a cyclist", "a child",
        "a teenager", "an elderly man", "an elderly woman", "an adult", "a customer",
        "a patron", "a friend", "a neighbor", "a parent", "a sibling", "a bystander",
        "a pedestrian", "the person in the frame", "a person nearby", "a cook", "a chef",
        "a nurse", "a doctor", "a teacher", "a youngster", "a kid"
    ]
    random.shuffle(replacements_pool)
    pool_iter = iter(replacements_pool)
    label_map = {}
    pattern = re.compile(r'#([A-Za-z0-9]+)\s+\1\b')

    def choose_replacement(label):
        if label in label_map:
            return label_map[label]
        try:
            choice = next(pool_iter)
        except StopIteration:
            choice = random.choice(replacements_pool)
        label_map[label] = choice
        return choice

    def replace_labels_in_caption(caption):
        def repl(match):
            label = match.group(1)
            choice = choose_replacement(label)
            start_index = match.start()
            needs_cap = (
                start_index == 0 or
                re.search(r'[.!?]\s*$', caption[:start_index]) is not None
            )
            if needs_cap:
                choice = choice[0].upper() + choice[1:]
            return choice

        out = caption
        while True:
            m = pattern.search(out)
            if not m:
                break
            out = out[:m.start()] + repl(m) + out[m.end():]
        return out

    return [replace_labels_in_caption(c) for c in captions]


# Example usage:
batch = [
    "#C C opens the door.",
    "#C C picks up a bag.",
    "#X X waves at #C C.",
    "Then #X X walks away."
]

for original, cleaned in zip(batch, replace_ego4d_labels_batch(batch)):
    print(f"Original: {original}")
    print(f"Cleaned:  {cleaned}")
    print()


if __name__ == "__main__":
    import pandas as pd

    data_dir = "/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI"
    csv_path = f"{data_dir}/ego4d-422K.csv"
    df = pd.read_csv(csv_path)
    
    from tqdm import tqdm
    
    # tqdm.pandas(desc="Cleaning dataframe")
    # df['sent0'] = df['sent0'].progress_apply(replace_ego4d_labels)
    # df['sent1'] = df['sent1'].progress_apply(replace_ego4d_labels)
    # df['hard_neg'] = df['hard_neg'].progress_apply(replace_ego4d_labels)
    
    # df[['sent0', 'sent1', 'hard_neg']] = df[['sent0', 'sent1', 'hard_neg']].progress_apply(replace_ego4d_labels_batch, axis=1)
    iterator = tqdm(range(len(df)), desc="Cleaning dataframe")
    df_clean = []
    for i in iterator:
        sentences = [df.loc[i, 'sent0'], df.loc[i, 'sent1'], df.loc[i, 'hard_neg']]
        sentences = replace_ego4d_labels_batch(sentences)
        df_clean.append(sentences)
    df_clean = pd.DataFrame(df_clean, columns=['sent0', 'sent1', 'hard_neg'])
    df_clean.to_csv(f"{data_dir}/ego4d-422K-cleaned.csv", index=False)
    print(f"Saved the cleaned dataframe to {data_dir}/ego4d-422K-cleaned.csv")
    