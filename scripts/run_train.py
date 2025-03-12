import argparse
import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import JSONLDataset
from models.transformer import TransformerModel
from tqdm import tqdm
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def train(args):
    print(f"Using device: {args.device}")
    print(f"Training with data from: {args.train_data}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = JSONLDataset(args.train_data, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Number of training examples: {len(dataset)}")
    
    # Initialize model with basic config
    model_config = {
        'n_layers': 6,
        'n_heads': 8,
        'dim': 512,
        'hidden_dim': 2048,
        'dropout': 0.1
    }
    
    # Initialize model
    model = TransformerModel(
        vocab_size=dataset.vocab_size,
        max_length=args.max_length,
        **model_config
    ).to(args.device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Initialize wandb
    wandb.init(
        project="text-diffusion",
        config={
            **vars(args),
            **model_config
        }
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for batch in pbar:
                batch = batch.to(args.device)
                
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                wandb.log({
                    'loss': loss.item(),
                    'epoch': epoch
                })
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': {
                    'char_to_idx': dataset.char_to_idx,
                    'idx_to_char': dataset.idx_to_char
                },
                'config': model_config,
                'avg_loss': avg_loss
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')

if __name__ == '__main__':
    args = parse_args()
    train(args)