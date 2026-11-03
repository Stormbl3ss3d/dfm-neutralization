import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class WNCDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load the processed JSONL data
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # 1. Format the string
        # We use a simple structural prompt
        prompt_text = f"Biased: {row['src']} Neutral:"
        target_text = f" {row['tgt']}"
        
        # 2. Tokenize separately to find the boundary
        prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_enc = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        
        # 3. Combine with special tokens: [CLS] prompt [SEP] target [SEP]
        input_ids = (
            [self.tokenizer.cls_token_id] + 
            prompt_enc + 
            [self.tokenizer.sep_token_id] + 
            target_enc + 
            [self.tokenizer.sep_token_id]
        )
        
        # Calculate exactly where the target starts and ends
        target_start_idx = 1 + len(prompt_enc) + 1 # +1 for CLS, +1 for first SEP
        target_end_idx = target_start_idx + len(target_enc)
        
        # 4. Truncate if necessary (keeping the final SEP if possible)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length-1] + [self.tokenizer.sep_token_id]
            target_end_idx = min(target_end_idx, self.max_length - 1)
            
        # 5. Create the target mask (1 for tokens we want to generate, 0 for prompt/padding)
        target_mask = [0] * self.max_length
        for i in range(target_start_idx, target_end_idx):
            target_mask[i] = 1
            
        # 6. Pad the input_ids
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool)
        }

def get_wnc_dataloaders(train_path, dev_path, tokenizer_name="distilroberta-base", batch_size=32, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    train_dataset = WNCDataset(train_path, tokenizer, max_length)
    dev_dataset = WNCDataset(dev_path, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader, tokenizer

# Quick test to ensure it works when run directly
if __name__ == "__main__":
    import os
    train_file = os.path.expanduser("~/dfm-neutralization/results/infill_baseline.jsonl") # Using the test file just as a quick sanity check path
    train_loader, dev_loader, tok = get_wnc_dataloaders(train_file, train_file, batch_size=2)
    batch = next(iter(train_loader))
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Target Mask shape:", batch["target_mask"].shape)
    
    # Print the decoded text for the first item to verify our boundary logic
    ids = batch["input_ids"][0]
    mask = batch["target_mask"][0]
    print("\nFull Decoded text:")
    print(tok.decode(ids[ids != tok.pad_token_id]))
    print("\nMasked Target portion only:")
    print(tok.decode(ids[mask]))
