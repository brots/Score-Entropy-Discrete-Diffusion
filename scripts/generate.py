import argparse
import torch
from pathlib import Path
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import TransformerModel

def generate_text(model, vocab, device, max_length=128, temperature=1.0):
    model.eval()
    # Start with a single token (using index 1 as start token)
    current_seq = torch.ones((1, 1), dtype=torch.long, device=device)
    
    generated_indices = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model's output logits
            embeddings = model.embedding(current_seq) + model.pos_encoding[:, :current_seq.size(1), :]
            
            # Create attention mask for the transformer
            # We want to attend to all previous tokens
            attn_mask = torch.zeros((current_seq.size(1), current_seq.size(1)), device=device).bool()
            
            hidden_states = model.transformer(embeddings, mask=attn_mask)
            logits = model.output_layer(hidden_states)
            
            # Get next token probabilities from the last position
            next_token_logits = logits[0, -1, :] / temperature
            probabilities = torch.softmax(next_token_logits, dim=0)
            
            # Sample from the distribution
            next_token = torch.multinomial(probabilities, 1)
            
            # Add to generated sequence
            generated_indices.append(next_token.item())
            
            # Update input sequence
            current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
            
            # Optional: stop if we generate an end token (if you have one)
            # if next_token.item() == end_token_id:
            #     break
    
    # Convert indices to text
    try:
        generated_text = ''.join(vocab['idx_to_char'][str(idx)] for idx in generated_indices)
    except KeyError:
        generated_text = ''.join(vocab['idx_to_char'][idx] for idx in generated_indices)
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (higher = more random)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Initialize model with saved config
    model = TransformerModel(
        vocab_size=len(checkpoint['vocab']['idx_to_char']),
        max_length=args.max_length,
        **checkpoint['config']
    ).to(args.device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully. Last epoch: {checkpoint['epoch']}, Loss: {checkpoint['avg_loss']:.4f}")
    
    # Print vocabulary information
    print(f"Vocabulary size: {len(checkpoint['vocab']['idx_to_char'])}")
    
    # Generate samples
    print("\nGenerating samples:\n" + "="*50)
    for i in range(args.num_samples):
        generated_text = generate_text(
            model, 
            checkpoint['vocab'], 
            args.device,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"\nSample {i+1}:")
        print(f"{generated_text}")
        print("="*50)

if __name__ == '__main__':
    main() 