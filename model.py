from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BERTArgs:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 30522  # Default vocab size for BERT
    max_seq_len: int = 512
    num_classes: int = 5
    dropout_prob: float = 0.1
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

class Embeddings(nn.Module):
    def __init__(self, args= BERTArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.position_embeddings = nn.Embedding(args.max_seq_len, args.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, args.hidden_size)

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(input_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadAttention(nn.Module):

    def __init__(self, args= BERTArgs) -> None:
        super().__init__()
        self.num_heads = args.num_heads
        self.head_dim = args.hidden_size // self.num_heads
        assert args.hidden_size % self.num_heads == 0, "Hidden size must be divisible by the number of heads"

        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout_prob)
        # Attention output layer
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
    
    def forward(self, hidden_states, attention_mask=None):
        # Get queries, keys, values
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Scale queries and perform attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Get context from values
        context_layer = torch.matmul(attention_probs, value_layer)
        output_layer = self.dense(context_layer)
        output_layer = self.LayerNorm(hidden_states + output_layer)
        return output_layer

class FeedForward(nn.Module):

    def __init__(self, args=BERTArgs):
        super().__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.dense_2 = nn.Linear(args.intermediate_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
    
    def forward(self, hidden_states):
        intermediate_output = self.dense_1(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        intermediate_output = self.dense_2(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        return self.LayerNorm(hidden_states + intermediate_output)

class BERTEncoderLayer(nn.Module):

    def __init__(self, args:BERTArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
    
    def forward(self, hidden_states, mask=None):
        attention = self.attention(hidden_states, mask)
        out = self.feed_forward(attention)
        return out

class BERTEncoder(nn.Module):

    def __init__(self, args= BERTArgs):
        super().__init__()
        self.layers = nn.ModuleList([BERTEncoderLayer(args) for _ in range(args.num_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class BERTClassification(nn.Module):

    def __init__(self, args = BERTArgs):
        super().__init__()
        self.embeddings = Embeddings(args)
        self.enoder = BERTEncoder(args)
        self.pooler = nn.Linear(args.hidden_size, args.hidden_size)
        self.classifier = nn.Linear(args.hidden_size, args.num_classes)
        self.dropout = nn.Dropout(args.dropout_prob)
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding layer
        embeddings = self.embeddings(input_ids)
        
        # Encoder
        encoder_output = self.encoder(embeddings, attention_mask)

        # Pooler: Take [CLS] token output (first token)
        cls_token_output = encoder_output[:, 0]
        pooled_output = F.tanh(self.pooler(cls_token_output))

        # Classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits