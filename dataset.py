import torch
from torch.utils.data import Dataset


class TokenizedMovieDataset(Dataset):
    def __init__(self, all_movies, tokenizer):
        self.tokenizer = tokenizer
        self.data = []

        for movie_idx in range(len(all_movies['text'])):
            lines = all_movies['text'][movie_idx]
            speakers = all_movies['speaker_ids'][movie_idx]

            # Build full text and per-character speaker ID list
            full_text = ""
            full_speaker_ids = []

            for line, speaker in zip(lines, speakers):
                full_text += line
                full_speaker_ids.extend([speaker] * len(line))

            # Tokenize with offset mapping
            encoded = tokenizer(
                full_text,
                return_offsets_mapping=True,
                return_tensors=None,
                add_special_tokens=False  # optional, depending on use
            )

            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]

            # Map each token to a speaker ID based on the token's starting char
            token_speaker_ids = []
            for start, end in offsets:
                if start < len(full_speaker_ids):
                    token_speaker_ids.append(full_speaker_ids[start])
                else:
                    # In case offset is beyond the text (shouldn't happen, but safe fallback)
                    token_speaker_ids.append("UNKNOWN")

            self.data.append({
                "input_ids": input_ids,
                "token_speaker_ids": token_speaker_ids
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]