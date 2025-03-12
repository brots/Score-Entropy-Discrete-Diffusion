import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, max_length, n_layers=6, n_heads=8, dim=512, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.output_layer = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeddings = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        
        # Create attention mask to prevent attention to padding tokens
        mask = x == 0  # Assuming 0 is the padding token
        
        # Transform the data
        hidden_states = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # Get logits
        logits = self.output_layer(hidden_states)
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), ignore_index=0)
        
        return loss 