import os
import os.path
import yaml
import oxen

import torch
import yaml


import argparse
from sedd.datasets.ox_dataset import OxDataset
from sedd.datasets.brown_cow_dataset import BrownCowDataset
from sedd.datasets.wikitext2_dataset import Wikitext2Dataset
from sedd.datasets.open_subtitles_dataset import OpenSubtitlesDataset
from sedd.datasets.baby_names_dataset import BabyNamesDataset
from sedd.datasets.abc_dataset import ABCDataset

from sedd.tokenizers.ox_tokenizer import OxTokenizer
from sedd.tokenizers.abc_tokenizer import ABCTokenizer
from sedd.models.noise import LogLinearNoise
from sedd.models.sedd import SEDD
from sedd.models.sampler import Sampler
from sedd.models.graph import AbsorbingGraph
from sedd.trainer.trainer import Trainer
from sedd.eval.evaluator import Evaluator
from transformers import GPT2TokenizerFast

from aim import Run

# from sedd.models.simple_sedd import SEDD
from torch.utils.data import DataLoader
from pathlib import Path
from data.datasets import JSONLDataset
from models.transformer import TransformerModel
from tqdm import tqdm
import wandb

def print_devices(device):
    if torch.cuda.is_available():
        print("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        print("WARNING: Using device {}".format(device))
    print(f"Using device: {device}")
    print(f"Found {os.cpu_count()} total number of CPUs.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def train(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = JSONLDataset(args.train_data, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = TransformerModel(
        vocab_size=dataset.vocab_size,
        max_length=args.max_length,
        **config['model']
    ).to(args.device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Initialize wandb
    wandb.init(
        project="text-diffusion",
        config={
            **vars(args),
            **config
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
                }
            }, checkpoint_path)
            
        print(f'Epoch {epoch+1} average loss: {total_loss/len(dataloader):.4f}')

def main():
    args = argparse.ArgumentParser(description="Train SEDD")
    args.add_argument("--cfg", type=str, default="configs/config.yaml")
    args.add_argument("--output", type=str, default="output")
    args.add_argument("--repo", type=str, default="ox/SEDD_dev")
    args = args.parse_args()

    # load in tokenizer
    # tokenizer = OxTokenizer()
    tokenizer = ABCTokenizer()
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    print("Got EOS token: ", tokenizer.eos_token)
    tokenizer.pad_token = '' # make sure we pad with absorbing token

    with open(args.cfg, 'r') as f:
        cfg = yaml.full_load(f)

    cfg['tokens'] = tokenizer.vocab_size
    cfg['data'] = {}
    cfg['data']['remote_repo'] = args.repo
    cfg['training']['output_dir'] = args.output

    print(cfg)

    work_dir = cfg['training']['output_dir']

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    print(work_dir)
    print(cfg)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print_devices(device)

    # Create remote oxen repo
    repo = oxen.RemoteRepo(cfg['data']['remote_repo'])
    if not repo.exists():
        repo.create()

    # Save config file for this run
    repo.add('configs/config.yaml')
    repo.commit("Added config file")

    # build token graph
    graph = AbsorbingGraph(tokenizer.vocab_size)

    # build score model
    score_model = SEDD(cfg, tokenizer.vocab_size).to(device)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    print(f"Number of parameters in the model: {num_parameters}")

    # train_ds = DataLoader(OpenSubtitlesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=10_000), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    # eval_ds = DataLoader(OpenSubtitlesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128))

    train_ds = DataLoader(BabyNamesDataset(tokenizer, seq_len=cfg['model']['length']), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    eval_ds = DataLoader(BabyNamesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128, train=False))
    
    # train_ds = DataLoader(ABCDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=10000), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    # eval_ds = DataLoader(ABCDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128))

    noise = LogLinearNoise().to(device)

    run = Run()
    run["hparams"] = cfg

    def eval(state):
        evaluator = Evaluator(eval_ds, run, cfg, device=device)
        return evaluator.evaluate(state)

    def sample(state):
        step = state['step']
        model = state['model']
        graph = state['graph']
        noise = state['noise']

        sampler = Sampler(cfg)
        texts = sampler.sample(tokenizer, model, graph, noise, steps=128, batch_size=cfg['eval']['batch_size'])

        file_name = os.path.join(sample_dir, f"sample.txt")
        with open(file_name, 'w') as file:
            for sentence in texts:
                file.write(sentence + "\n")
                file.write("="*80 + "\n")

        # Push samples to Oxen.ai for tracking
        repo = oxen.RemoteRepo(cfg['data']['remote_repo'])
        repo.add(file_name)
        repo.commit(f"Sample at step {step}")

    trainer = Trainer(
        run,
        score_model,
        graph,
        noise,
        cfg,
        eval_callback=eval,
        sample_callback=sample,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    trainer.train(train_ds)

    # Parse command line arguments and train the model
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()