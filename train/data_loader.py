import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Optional, Union, Iterator
from pathlib import Path
import json
from tqdm import tqdm
import logging
from datasets import load_dataset
import random

logger = logging.getLogger(__name__)

class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large-scale text data"""
    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        split: str = "train",
        seq_len: int = 512,
        text_column: str = "text",
        buffer_size: int = 1000,
        cache_dir: Optional[str] = None,
        config: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.buffer_size = buffer_size
        
        # Load dataset
        self.dataset = load_dataset(
            dataset_name,
            config,
            split=split,
            streaming=True,
            cache_dir=cache_dir
        )
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        buffer = []
        current_size = 0
        
        iterator = iter(self.dataset)
        while True:
            # Fill buffer if needed
            if current_size < self.seq_len + 1:
                try:
                    item = next(iterator)
                    text = item[self.text_column]
                    if not text.strip():
                        continue
                    
                    # Tokenize
                    tokens = self.tokenizer.encode(text)
                    buffer.extend(tokens)
                    current_size = len(buffer)
                except StopIteration:
                    if current_size < self.seq_len + 1:
                        break
            
            # Yield a sample if we have enough tokens
            if current_size >= self.seq_len + 1:
                tokens = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                current_size = len(buffer)
                
                x = torch.tensor(tokens[:-1], dtype=torch.long)
                y = torch.tensor(tokens[1:], dtype=torch.long)
                yield x, y

class TextDataset(Dataset):
    """Dataset for smaller text corpora that fit in memory"""
    def __init__(
        self,
        tokenizer,
        texts: Union[List[str], str],
        seq_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Handle both file paths and direct text input
        if isinstance(texts, str) and Path(texts).exists():
            logger.info(f"Loading texts from {texts}")
            with open(texts, 'r', encoding='utf-8') as f:
                if texts.endswith('.json'):
                    texts = [item['text'] for item in json.load(f)]
                else:
                    texts = f.readlines()
        
        # Tokenize all texts
        logger.info("Tokenizing texts...")
        tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            if not text.strip():
                continue
            tokens.extend(self.tokenizer.encode(text))
            tokens.append(self.tokenizer.token_to_id['</W>'])  # Add EOS token
            
        # Create training samples
        self.samples = []
        for i in range(0, len(tokens) - seq_len):
            chunk = tokens[i:i + seq_len + 1]
            self.samples.append(chunk)
            
        logger.info(f"Created {len(self.samples)} training samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloader(
    tokenizer,
    source: Union[str, List[str]],
    seq_len: int = 512,
    batch_size: int = 32,
    shuffle: bool = True,
    streaming: bool = False,
    config: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create a dataloader for training.
    
    Args:
        tokenizer: Tokenizer instance
        source: Either a dataset name (for streaming) or path/list of texts
        seq_len: Sequence length for training
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        streaming: Whether to use streaming dataset
        config: Dataset configuration name (for HuggingFace datasets)
        **kwargs: Additional arguments passed to the dataset
    """
    if streaming:
        dataset = StreamingTextDataset(
            tokenizer=tokenizer,
            dataset_name=source,
            seq_len=seq_len,
            config=config,
            **kwargs
        )
        shuffle = False  # Streaming datasets can't be shuffled
    else:
        dataset = TextDataset(
            tokenizer=tokenizer,
            texts=source,
            seq_len=seq_len
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4 if not streaming else 0,
        pin_memory=True
    )

