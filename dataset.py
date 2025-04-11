import torch
from torch.utils.data import Dataset
import random


class TokenizedMovieDataset(Dataset):
    """
    A PyTorch Dataset class for processing movie dialogue data.
    
    This class tokenizes movie dialogue, maps speaker IDs to sequential integers,
    and creates data points for each position in each movie. It returns input tokens,
    labels, speaker masks, and attention masks for model training and evaluation.
    
    Attributes:
        tokenizer: The tokenizer used to process text
        data: List of dictionaries containing tokenized movie data
        padding: Whether to pad sequences
        max_length: Maximum sequence length
        min_length: Minimum sequence length for a movie to be included
        speaker_to_id: Mapping from speaker names to sequential IDs
        id_to_speaker: Mapping from sequential IDs to speaker names
        index_mapping: List of (movie_idx, position) pairs for data points
    """
    
    def __init__(self, all_movies, tokenizer, padding=False, max_length=512, min_length=100):
        """
        Initialize the dataset with tokenized movie data.
        
        Args:
            all_movies (dict): Dictionary containing movie text and speaker IDs
            tokenizer: The tokenizer to use for processing text
            padding (bool, optional): Whether to pad sequences. Defaults to False.
            max_length (int, optional): Maximum sequence length. Defaults to 512.
            min_length (int, optional): Minimum sequence length for a movie. Defaults to 100.
        """
        self.tokenizer = tokenizer
        self.data = []
        self.padding = padding
        self.max_length = max_length
        self.min_length = min_length
        
        # Create a mapping of unique speaker IDs to integers
        self.speaker_to_id = {}
        self.id_to_speaker = {}
        current_id = 0

        for movie_idx in range(len(all_movies['text'])):
            lines = all_movies['text'][movie_idx]
            speakers = all_movies['speaker_ids'][movie_idx]

            # Build full text and per-character speaker ID list
            full_text = ""
            full_speaker_ids = []

            for line, speaker in zip(lines, speakers):
                full_text += line
                full_speaker_ids.extend([speaker] * len(line))
                
            encoded = tokenizer(
                    full_text,
                    return_offsets_mapping=True,
                    return_tensors=None,
                    add_special_tokens=False,  # optional, depending on use
                    max_length=max_length,  # Truncate to max_length
                    truncation=True
                )
                
            input_ids = encoded["input_ids"]
            if len(input_ids) < self.min_length:
                continue
            offsets = encoded["offset_mapping"]

            # Map each token to a speaker ID based on the token's starting char
            token_speaker_ids = []
            # Keep track of speakers we've seen in this movie
            speaker_order = {}
            next_id = 0
            
            for start, end in offsets:
                if start < len(full_speaker_ids):
                    speaker = full_speaker_ids[start]
                    # Assign sequential IDs as we encounter new speakers
                    if speaker not in speaker_order:
                        speaker_order[speaker] = next_id
                        next_id += 1
                    token_speaker_ids.append(speaker_order[speaker])
                else:
                    # In case offset is beyond the text (shouldn't happen, but safe fallback)
                    token_speaker_ids.append(0)

            # Convert input_ids and token_speaker_ids to tensors
            input_ids = torch.tensor(input_ids, dtype=torch.float16)
            token_speaker_ids = torch.tensor(token_speaker_ids, dtype=torch.float16)
            
            self.data.append({
                "input_ids": input_ids,
                "token_speaker_ids": token_speaker_ids
            })
            
        # Create a mapping from global index to (movie_idx, position)
        self.index_mapping = []
        for movie_idx, movie_data in enumerate(self.data):
            seq_length = len(movie_data["input_ids"])
            # For each movie, create data points for all positions from min_length to seq_length-1
            for position in range(self.min_length - 1, seq_length - 1):
                self.index_mapping.append((movie_idx, position))
                
        print(f"Created {len(self.index_mapping)} total data points from {len(self.data)} movies")

    def __len__(self):
        """
        Return the total number of data points in the dataset.
        
        Returns:
            int: The number of data points
        """
        return len(self.index_mapping)

    def getitem_old(self, idx):
        """
        Legacy method to get a movie by index.
        
        Args:
            idx (int): Index of the movie
            
        Returns:
            dict: Movie data containing input_ids and token_speaker_ids
        """
        return self.data[idx]

    def getitem_doc_with_position(self, doc_idx, position):
        """
        Get tokens up to a specific position in a document, along with the next token as label
        and a speaker mask indicating which previous tokens share the same speaker as the label.
        
        Args:
            doc_idx (int): Index of the document in the dataset
            position (int): Number of tokens to include (0-based)
            
        Returns:
            dict: Contains:
                - input_ids: Tensor of tokens up to position, padded to max_length
                - label: The next token after position
                - speaker_mask: Binary mask indicating which previous tokens share speaker with label
                - attention_mask: Binary mask indicating which tokens are actual content (1) vs padding (0)
        """
        doc = self.data[doc_idx]
        input_ids = doc["input_ids"]
        speaker_ids = doc["token_speaker_ids"]
        
        # Ensure position is valid
        if position >= len(input_ids) - 1:
            raise ValueError(f"Position {position} is too large for document of length {len(input_ids)}")
            
        # Get tokens up to position
        context_tokens = input_ids[:position + 1]
        context_speaker_ids = speaker_ids[:position + 1]
        
        # Create attention mask (1 for actual tokens, 0 for padding)
        attention_mask = torch.ones(len(context_tokens), dtype=torch.float16)
        
        # Pad or truncate to max_length (truncate from the left to keep the most recent context)
        if len(context_tokens) > self.max_length:
            context_tokens = context_tokens[-self.max_length:]
            context_speaker_ids = context_speaker_ids[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
        elif len(context_tokens) < self.max_length:
            # Pad with zeros (assuming 0 is the padding token)
            padding = torch.zeros(self.max_length - len(context_tokens), dtype=torch.float16)
            context_tokens = torch.cat([padding, context_tokens])
            # Also pad speaker_ids
            speaker_padding = torch.zeros(self.max_length - len(context_speaker_ids), dtype=torch.float16)
            context_speaker_ids = torch.cat([speaker_padding, context_speaker_ids])
            # Pad attention mask with zeros
            attention_padding = torch.zeros(self.max_length - len(attention_mask), dtype=torch.float16)
            attention_mask = torch.cat([attention_padding, attention_mask])
        
        # Get the next token as label
        label = input_ids[position + 1]
        # Get speaker ID of the label token
        label_speaker = speaker_ids[position + 1]
        # Create binary mask for same speaker
        speaker_mask = (context_speaker_ids == label_speaker).float()
        
        return {
            "input_ids": context_tokens,
            "label": label,
            "speaker_mask": speaker_mask,
            "attention_mask": attention_mask
        }
        
    def __getitem__(self, idx):
        """
        Main entry point for PyTorch DataLoader.
        
        Args:
            idx (int or slice): Index or slice of items to retrieve
            
        Returns:
            dict or list: Data point(s) containing input_ids, label, speaker_mask, and attention_mask
        """
        # Handle slicing
        if isinstance(idx, slice):
            # Create a new dataset with the sliced indices
            sliced_dataset = TokenizedMovieDataset.__new__(TokenizedMovieDataset)
            sliced_dataset.tokenizer = self.tokenizer
            sliced_dataset.data = self.data
            sliced_dataset.padding = self.padding
            sliced_dataset.max_length = self.max_length
            sliced_dataset.min_length = self.min_length
            sliced_dataset.speaker_to_id = self.speaker_to_id
            sliced_dataset.id_to_speaker = self.id_to_speaker
            
            # Slice the index_mapping
            sliced_dataset.index_mapping = self.index_mapping[idx]
            
            return sliced_dataset
        
        # Get the movie index and position from the mapping
        movie_idx, position = self.index_mapping[idx]
        return self.getitem_doc_with_position(movie_idx, position)

class TokenizedMovieDatasetForMaskedLM(TokenizedMovieDataset):
    def __init__(self, all_movies, tokenizer, padding=False, max_length=512, min_length=100):
        super().__init__(all_movies, tokenizer, padding, max_length, min_length)
        
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        #modify input_ids
        input_ids = item["input_ids"]
        input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.mask_token_id])])
        #create new labels
        label = input_ids.clone()
        label[-1] = item['label']
        label[:-1] = -100
        # modify speaker mask
        speaker_mask = item["speaker_mask"]
        speaker_mask = torch.cat([speaker_mask, torch.tensor([1])])
        return {"input_ids": input_ids, "label": label, "speaker_mask":  speaker_mask}
    
    


class RandomSubsetDataset(Dataset):
    """
    A dataset wrapper that returns a random subset of another dataset.
    
    This class is useful for testing and evaluation purposes, allowing you to
    quickly get a random subset of a larger dataset without modifying the original.
    
    Attributes:
        dataset: The original dataset to sample from
        indices: The indices of the items to include in the subset
    """
    
    def __init__(self, dataset, size=None, fraction=None, seed=None):
        """
        Initialize the random subset dataset.
        
        Args:
            dataset: The original dataset to sample from
            size (int, optional): The number of items to include in the subset.
                                If None, fraction is used instead.
            fraction (float, optional): The fraction of the dataset to include.
                                      Used only if size is None.
            seed (int, optional): Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
            
        if size is not None:
            self.size = min(size, len(dataset))
        elif fraction is not None:
            self.size = int(len(dataset) * fraction)
        else:
            raise ValueError("Either size or fraction must be provided")
            
        self.dataset = dataset
        self.indices = random.sample(range(len(dataset)), self.size)
        
    def __len__(self):
        """
        Return the number of items in the subset.
        
        Returns:
            int: The number of items in the subset
        """
        return self.size
        
    def __getitem__(self, idx):
        """
        Get an item from the subset.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            The item from the original dataset at the randomly selected index
        """
        return self.dataset[self.indices[idx]]
        
    def get_indices(self):
        """
        Get the indices of the items in the subset.
        
        Returns:
            list: The indices of the items in the subset
        """
        return self.indices