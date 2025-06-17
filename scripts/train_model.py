import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import TransformerModel
from tokenizer.bpe import BPETokenizer
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pathlib import Path
import json
import argparse
from train.data_loader import get_dataloader
import logging
from tqdm import tqdm
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_dataloader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, path)
    return path

def load_checkpoint(path, model, optimizer, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def train(model, train_dataloader, val_dataloader, optimizer, device, epochs=1, 
          grad_clip=1.0, scheduler=None, log_wandb=False, checkpoint_dir='checkpoints',
          checkpoint_freq=1, resume_from=None, steps_per_epoch=1000,
          gradient_accumulation_steps=1):
    
    model.train()
    best_val_loss = float('inf')
    start_epoch = 0
    global_step = 0
    
    if resume_from:
        start_epoch, _ = load_checkpoint(resume_from, model, optimizer, scheduler)
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Try to initialize wandb, but don't fail if it's not configured
    wandb_run = None
    if log_wandb:
        try:
            wandb_run = wandb.init(
                project="nanollm",
                config={
                    "epochs": epochs,
                    "batch_size": train_dataloader.batch_size if hasattr(train_dataloader, 'batch_size') else None,
                    "grad_clip": grad_clip,
                    "steps_per_epoch": steps_per_epoch,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_batch_size": train_dataloader.batch_size * gradient_accumulation_steps if hasattr(train_dataloader, 'batch_size') else None,
                }
            )
            logger.info("Successfully initialized wandb logging")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging...")
            log_wandb = False
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for step, batch in enumerate(train_dataloader):
            if step >= steps_per_epoch:
                break
                
            if isinstance(batch, (tuple, list)):
                x, y = batch
            else:
                x = batch['input_ids']
                y = batch['labels']
            
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Only update weights after accumulating enough gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * gradient_accumulation_steps  # Scale loss back for logging
            avg_loss = total_loss / (step + 1)
            pbar.update(1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            if (step + 1) % 50 == 0 and log_wandb and wandb_run is not None:
                wandb.log({
                    "train_loss": avg_loss,
                    "epoch": epoch,
                    "step": global_step,
                    "lr": optimizer.param_groups[0]["lr"]
                })
            
            global_step += 1
        
        pbar.close()
        
        # Validation
        if val_dataloader is not None:
            val_loss = 0
            val_steps = min(steps_per_epoch // 10, 100)  # Validate on 10% of training steps
            val_pbar = tqdm(total=val_steps, desc="Validating")
            
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(val_dataloader):
                    if step >= val_steps:
                        break
                        
                    if isinstance(batch, (tuple, list)):
                        x, y = batch
                    else:
                        x = batch['input_ids']
                        y = batch['labels']
                    
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_loss += loss.item()
                    val_pbar.update(1)
            
            val_pbar.close()
            model.train()
            
            val_loss = val_loss / val_steps
            logger.info(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")
            
            if log_wandb and wandb_run is not None:
                wandb.log({
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "global_step": global_step
                })
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, checkpoint_dir)
                
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss,
                                         os.path.join(checkpoint_dir, 'best'))
                logger.info(f"New best model saved to {best_path}")

    # Clean up wandb
    if wandb_run is not None:
        wandb_run.finish()

def main():
    parser = argparse.ArgumentParser(description='Train NanoLLM')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to config file')
    parser.add_argument('--dataset', type=str, default='wikitext', help='Dataset name or path')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
    
    # If using a HuggingFace dataset, we can stream it
    if args.dataset in ['wikitext', 'c4', 'pile']:
        logger.info(f"Loading streaming dataset: {args.dataset}")
        if args.dataset == 'wikitext':
            dataset_name = 'wikitext'
            dataset_config = 'wikitext-103-raw-v1'
            logger.info(f"Using wikitext config: {dataset_config}")
        elif args.dataset == 'c4':
            dataset_name = 'c4'
            dataset_config = 'en'
        else:
            dataset_name = 'the_pile'
            dataset_config = None
        
        train_dataloader, val_dataloader = get_dataloader(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            streaming=True
        )
    else:
        # Load local dataset
        train_dataloader, val_dataloader = get_dataloader(
            dataset_path=args.dataset,
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            streaming=False
        )
    
    # Initialize model
    logger.info("Initializing model...")
    model = TransformerModel(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        seq_len=config['seq_len'],
        dropout=config['dropout']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
        eps=config['optimizer']['eps']
    )
    
    # Initialize scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] * 1000,  # steps_per_epoch
        eta_min=config['scheduler']['min_lr']
    )
    
    # Start training
    logger.info("Starting training...")
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        epochs=config['epochs'],
        grad_clip=config['grad_clip'],
        scheduler=scheduler,
        log_wandb=args.wandb,
        checkpoint_dir=config['training']['checkpoint_dir'],
        checkpoint_freq=config['training']['checkpoint_freq'],
        resume_from=args.resume_from,
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
    )

if __name__ == "__main__":
    main()

