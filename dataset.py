import torch
from torch.utils.data import Dataset


class TokenizedMovieDataset(Dataset):
    def __init__(self, all_movies, tokenizer, padding=False, max_length=512):
        self.tokenizer = tokenizer
        self.data = []
        self.padding = padding
        self.max_length = max_length
        
        # Create a mapping of unique speaker IDs to integers
        self.speaker_to_id = {}
        self.id_to_speaker = {}
        current_id = 0
        
        # First pass: collect all unique speaker IDs
        for movie_idx in range(len(all_movies['text'])):
            speakers = all_movies['speaker_ids'][movie_idx]
            for speaker in speakers:
                if speaker not in self.speaker_to_id:
                    self.speaker_to_id[speaker] = current_id
                    self.id_to_speaker[current_id] = speaker
                    current_id += 1

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
            if not self.padding:
                encoded = tokenizer(
                    full_text,
                    return_offsets_mapping=True,
                    return_tensors=None,
                    add_special_tokens=False  # optional, depending on use
                )
            else:
                encoded = tokenizer(
                    full_text,
                    return_offsets_mapping=True,
                    return_tensors=None,
                    add_special_tokens=False,  # optional, depending on use
                    max_length=self.max_length,
                    padding='max_length',  # Add explicit padding parameter
                    truncation=True  # Truncate if too long
                )

            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]

            # Map each token to a speaker ID based on the token's starting char
            token_speaker_ids = []
            for start, end in offsets:
                if start < len(full_speaker_ids):
                    speaker = full_speaker_ids[start]
                    # Convert string speaker ID to numerical ID
                    speaker_id = self.speaker_to_id.get(speaker, self.speaker_to_id["UNKNOWN"])
                    token_speaker_ids.append(speaker_id)
                else:
                    # In case offset is beyond the text (shouldn't happen, but safe fallback)
                    token_speaker_ids.append(self.speaker_to_id["UNKNOWN"])

            # Convert input_ids and token_speaker_ids to tensors
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            token_speaker_ids = torch.tensor(token_speaker_ids, dtype=torch.long)
            
            self.data.append({
                "input_ids": input_ids,
                "token_speaker_ids": token_speaker_ids
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]