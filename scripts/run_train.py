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

def print_gpu_info():
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB\n")
        
        # Enable memory tracking for better monitoring
        torch.cuda.reset_peak_memory_stats()
    else:
        print("\nNo GPU available, using CPU\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')  # Increased default batch size
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def train(args):
    # Print GPU information
    print_gpu_info()
    print(f"Using device: {args.device}")
    print(f"Training with data from: {args.train_data}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = JSONLDataset(args.train_data, max_length=args.max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True if args.device=='cuda' else False,
        num_workers=4 if args.device=='cuda' else 0  # Enable multiple workers for faster data loading
    )
    
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Number of training examples: {len(dataset)}")
    
    # Initialize model with basic config
    model_config = {
        'n_layers': 2,        # Reduced from 6
        'n_heads': 2,         # Reduced from 8
        'dim': 32,           # Reduced from 512
        'hidden_dim': 64,    # Reduced from 2048
        'dropout': 0.1
    }
    
    # Initialize model and move to GPU if available
    model = TransformerModel(
        vocab_size=dataset.vocab_size,
        max_length=args.max_length,
        **model_config
    ).to(args.device)
    
    if args.device == 'cuda':
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
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
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for batch in pbar:
                # Move batch to GPU if available
                batch = batch.to(args.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                if torch.cuda.is_available():
                    current_gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
                else:
                    current_gpu_memory = 0
                    peak_gpu_memory = 0
                
                wandb.log({
                    'loss': loss.item(),
                    'epoch': epoch,
                    'gpu_memory_current': current_gpu_memory,
                    'gpu_memory_peak': peak_gpu_memory
                })
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
        
        # Print GPU memory usage after each epoch
        if torch.cuda.is_available():
            print(f'Current GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
            print(f'Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB')
        
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