import torch
from torch import nn

class RecurrentLM(nn.Module):
    def __init__(self,vocab,kind='lstm',emb=100,hidden=128,embedding_weights=None,freeze_embedding=False):
        super().__init__()
        # Part I / Part II random baseline: nn.Embedding initializes the table
        # randomly, and requires_grad=True below lets language-model training
        # learn these vectors jointly with the recurrent network. ID 0 is padding.
        self.embedding=nn.Embedding(vocab,emb,padding_idx=0)
        if embedding_weights is not None:
            if tuple(embedding_weights.shape) != tuple(self.embedding.weight.shape):
                raise ValueError(f'embedding weights shape {tuple(embedding_weights.shape)} does not match {tuple(self.embedding.weight.shape)}')
            with torch.no_grad(): self.embedding.weight.copy_(embedding_weights)
        # Assignment control: the random baseline learns its randomly initialized
        # table jointly; Part II supplied Skip-gram/GloVe tables can be frozen in
        # their entirety by setting freeze_embedding=True.
        self.embedding.weight.requires_grad_(not freeze_embedding)
        # Use a vanilla RNN or LSTM over token vectors. batch_first means [batch, time, feature].
        self.rnn=(nn.LSTM if kind=='lstm' else nn.RNN)(emb,hidden,batch_first=True)
        # Convert each hidden state into one score (logit) per vocabulary token.
        self.head=nn.Linear(hidden,vocab)
    def forward(self,x):
        # Training predicts a next token at every time position in the input window.
        return self.head(self.encode(x))
    def encode(self,x):
        # Embedding lookup followed by recurrent sequence processing; discard final cell state.
        z,_=self.rnn(self.embedding(x))
        return z
    def next_logits(self,x,lengths=None):
        # For one prediction per sequence, select its last real input position (if lengths supplied).
        z=self.encode(x)
        if lengths is None:
            last=z[:,-1]
        else:
            idx=torch.as_tensor(lengths,device=x.device,dtype=torch.long)-1
            last=z[torch.arange(z.shape[0],device=x.device),idx]
        return self.head(last)

class TransformerLM(nn.Module):
    def __init__(self,vocab,emb=100,hidden=128,heads=4,layers=2,max_len=256):
        super().__init__()
        # Token vectors are projected to the Transformer width and combined with position vectors.
        self.embedding=nn.Embedding(vocab,emb,padding_idx=0)
        self.project=nn.Linear(emb,hidden)
        self.position=nn.Embedding(max_len,hidden)
        # EncoderLayer supplies self-attention + feed-forward; the causal mask below
        # turns this stack into a decoder-only LM without encoder-decoder cross-attention.
        layer=nn.TransformerEncoderLayer(hidden,heads,dim_feedforward=4*hidden,dropout=0.1,batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,layers)
        self.head=nn.Linear(hidden,vocab)
        self.max_len=max_len
    def forward(self,x):
        # Produce vocabulary scores at each position, used by next-token training loss.
        return self.head(self.encode(x))
    def encode(self,x):
        # Reject inputs that exceed the positional embedding table.
        b,t=x.shape
        if t>self.max_len: raise ValueError('sequence exceeds max_len')
        pos=torch.arange(t,device=x.device)
        z=self.project(self.embedding(x))+self.position(pos)[None,:,:]
        # Block attention to future positions so predictions cannot see the answer token.
        mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),diagonal=1)
        return self.encoder(z,mask=mask)
    def next_logits(self,x,lengths=None):
        # Return only the vocabulary scores at each sequence's final real context position.
        z=self.encode(x)
        if lengths is None:
            last=z[:,-1]
        else:
            idx=torch.as_tensor(lengths,device=x.device,dtype=torch.long)-1
            last=z[torch.arange(z.shape[0],device=x.device),idx]
        return self.head(last)

def build_model(kind,vocab,emb=100,hidden=128,heads=4,layers=2,max_len=256,embedding_weights=None,freeze_embedding=False):
    # One factory keeps training, evaluation, and generation using identical model definitions.
    if kind in ('rnn','lstm'): return RecurrentLM(vocab,kind,emb,hidden,embedding_weights,freeze_embedding)
    if kind=='transformer': return TransformerLM(vocab,emb,hidden,heads,layers,max_len)
    raise ValueError(kind)
