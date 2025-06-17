import re 
from collections import defaultdict, Counter 
from typing import List, Tuple, Dict 
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class BPETokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

    def get_stats(self, corpus: List[List[str]]) -> Counter:
        pairs = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def merge_vocab(self, corpus: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        pattern = re.escape(' '.join(pair))
        merged_token = ''.join(pair)
        new_corpus = []
        for word in corpus:
            joined = ' '.join(word)
            joined = re.sub(pattern, merged_token, joined)
            new_corpus.append(joined.split())
        return new_corpus

    def build_vocab(self, texts: List[str]):
        logger.info("Processing texts for vocabulary building...")
        # Initialize with special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '</W>']
        self.token_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Process texts in chunks to avoid memory issues
        chunk_size = 100  # Process 100 texts at a time
        all_pairs = Counter()
        
        for i in tqdm(range(0, len(texts), chunk_size), desc="Building initial vocabulary"):
            chunk = texts[i:i + chunk_size]
            # Split each text into words and characters
            corpus = []
            for line in chunk:
                for word in line.strip().split():
                    if len(word) > 0:  # Skip empty words
                        corpus.append(['<BOS>'] + list(word) + ['</W>'])
            
            # Count character pairs in this chunk
            pairs = self.get_stats(corpus)
            all_pairs.update(pairs)
            
            # If we have enough pairs, start merging
            if len(all_pairs) > self.vocab_size * 2:
                break
        
        logger.info(f"Found {len(all_pairs)} initial pairs")
        
        # Start with characters as initial tokens
        initial_tokens = set(special_tokens)
        for pair in all_pairs:
            initial_tokens.add(pair[0])
            initial_tokens.add(pair[1])
        
        logger.info(f"Initial vocabulary size: {len(initial_tokens)}")
        
        # Add remaining tokens through merging
        vocab_size = len(initial_tokens)
        pbar = tqdm(total=min(self.vocab_size - vocab_size, len(all_pairs)), desc="Merging tokens")
        
        while vocab_size < self.vocab_size and all_pairs:
            if not all_pairs:
                break
            best = all_pairs.most_common(1)[0][0]
            merged_token = ''.join(best)
            
            if merged_token not in initial_tokens:
                initial_tokens.add(merged_token)
                self.merges.append(best)
                vocab_size += 1
                pbar.update(1)
            
            # Remove the pair and its counts
            del all_pairs[best]
        
        pbar.close()
        
        # Create final vocabulary
        logger.info("Creating final vocabulary mappings...")
        all_tokens = sorted(list(initial_tokens))
        self.token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        logger.info(f"Final vocabulary size: {len(self.token_to_id)}")

    def encode_word(self, word: str) -> List[str]:
        if not word:
            return []
        tokens = list(word)
        tokens.append('/W')
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if(tokens[i], tokens[i + 1]) == merge:
                    tokens[i:i + 2] = [''.join(merge)]
                else:
                    i += 1
        return tokens 

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        word_tokens = []
        for word in text.strip().split():
            tokens = ['<BOS>'] + self.encode_word(word)
            word_tokens.extend(self.token_to_id.get(t, self.token_to_id['<UNK>']) for t in tokens)
        return word_tokens

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token[i] for i in ids]
        text = ''
        for token in tokens:
            if token in ['<PAD>', '<UNK>', '<BOS>']:
                continue
            elif token == '</W>':
                text += ' '
            else:
                text += token
        return text.strip()

