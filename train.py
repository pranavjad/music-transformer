import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
import wandb
from tokenizer import vocab_size, pad_id

### Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Load Data
train_data = torch.load('train_data.pt')
valid_data = torch.load('valid_data.pt')

### Define Dataloader
def elapsedms_to_tokens(elapsed_ms):
        res = []
        max_shift = 1000
        if elapsed_ms >= max_shift:
            res.extend([128 + 128 + 100] * (elapsed_ms // max_shift))
            elapsed_ms %= max_shift
        remaining_shift_id = (elapsed_ms // 10)
        if remaining_shift_id > 0:
            res.append(128 + 128 + remaining_shift_id)
        return res

def get_batch(batch_size):
    # sample batch
    idxs = torch.randint(0, train_data.shape[0] - (ctx_len + 1), (batch_size,1))
    idxs = idxs + torch.arange(ctx_len + 1)
    batch = train_data[idxs]

    # random augmentation
    pitch_shifts = np.arange(-3, 4)
    time_stretches = np.array([0.95, 0.975, 1.0, 1.025, 1.05])
    shifts = torch.tensor(np.random.choice(pitch_shifts, batch.shape[0])).view(-1, 1).expand_as(batch)
    stretches = np.random.choice(time_stretches, batch.shape[0])
    
    # shift
    note_mask = (batch >= 1) & (batch <= 128 + 128)
    shifted_data = batch + shifts.masked_fill(~note_mask, 0)
    
    # stretch
    stretched_data = []
    for i, seq in enumerate(shifted_data):
        elapsed_ms = 0
        tok_cnt = []
        stretched_seq = []
        for idx in seq.tolist():
            if (1 + 128 + 128 <= idx < 128 + 128 + 100): # time shift
                elapsed_ms += (idx - (1 + 128 + 128) + 1) * 10
                tok_cnt.append(idx)
            else:
                stretched_seq += elapsedms_to_tokens(math.ceil(elapsed_ms * stretches[i]))
                stretched_seq.append(idx)
                elapsed_ms = 0
                tok_cnt = []
        stretched_seq += elapsedms_to_tokens(elapsed_ms)
        stretched_data.append(torch.tensor(stretched_seq))

    # pad
    aug_batch = nn.utils.rnn.pad_sequence(stretched_data, batch_first=True, padding_value=pad_id)[:, :ctx_len + 1]
    return aug_batch.to(device)

### Decoder-only Transformer With Relative Global Attention
n_emb = 256
n_layers = 6
n_heads = 8
dropout = 0.1
head_size = 256
filter_size = 1024
ctx_len = 2048

device = torch.device("cuda:0")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, filter_size)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(filter_size, n_emb)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.relu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MultiHeadRelativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_emb, head_size * 3, bias=False)
        
        # positional embeddings
        self.pos_embs = nn.Parameter(torch.randn(n_heads, ctx_len, head_size // n_heads) * 0.02)
        
        # attention mask
        tril = torch.tril(torch.ones(ctx_len, ctx_len))
        self.register_buffer('causal_mask', torch.zeros(ctx_len, ctx_len, requires_grad=False).masked_fill(tril == 0, -1e9))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(head_size, n_emb)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        K, Q, V = self.c_attn(x).split(head_size, dim=-1)
        K = K.view(B, T, n_heads, head_size // n_heads).transpose(1, 2)
        Q = Q.view(B, T, n_heads, head_size // n_heads).transpose(1, 2)
        V = V.view(B, T, n_heads, head_size // n_heads).transpose(1, 2)
        
        # S_rel calculation
        S_rel = (Q @ self.pos_embs[:, -T:, :].mT)
        S_rel = S_rel * torch.tril(torch.ones(T, T)).flip(1).to(device) # mask out-of-bounds positions
        S_rel = torch.cat((torch.zeros(B, n_heads, T, 1).to(device), S_rel), dim=-1).reshape(B, n_heads, T + 1, T) # pad and reshape
        S_rel = S_rel[..., 1:, :] # remove first row

        attn_weights = ((Q @ K.mT) + S_rel) / math.sqrt(Q.shape[-1])
        attn_weights = attn_weights + self.causal_mask[:T, :T]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        if mask is not None: # padding mask
            attn_weights = attn_weights * mask
        attn_weights = self.attn_dropout(attn_weights)
        out = attn_weights @ V
        out = out.transpose(1, 2).reshape(B, T, head_size)
        out = self.resid_dropout(self.c_proj(out))
        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mhra = MultiHeadRelativeAttention()
        self.mlp = MLP()
        self.ln_1 = nn.LayerNorm(n_emb)
        self.ln_2 = nn.LayerNorm(n_emb)
    
    def forward(self, x, mask=None):
        x = x + self.mhra(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x

def create_padding_mask(batch):
    B, T = batch.shape
    # 0s in mask positions, 1s in other positions
    mask = torch.eq(batch, pad_id).unsqueeze(1)
    mask = ~(mask | mask.mT)
    mask = mask.view(B, 1, T, T)
    return mask.float().to(device)

class MusicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_emb)

        factor = 10000 ** (-torch.arange(0, n_emb, 2) / n_emb)
        pos = torch.arange(ctx_len).unsqueeze(1)
        sin_in = pos * factor
        pe = torch.zeros(ctx_len, n_emb)
        pe[:,0::2] = torch.sin(sin_in)
        pe[:,1::2] = torch.cos(sin_in)
        self.wpe = pe.to(device)
        self.wpe.requires_grad = False

        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_1 = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.input_dropout = nn.Dropout(dropout)
        param_cnt = sum([p.numel() for p in self.parameters()])
        print(f'{param_cnt/1e6:.3f}M parameters')
    
    def forward(self, x, mask=None):
        B, T = x.shape
        mask = create_padding_mask(x)
        tok_emb = self.wte(x)
        tok_emb *= math.sqrt(n_emb)
        pos_emb = self.wpe[:T]
        x = tok_emb + pos_emb
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_1(x)
        logits = self.head(x)
        return logits
    
    def loss(self, x, y):
        mask = (y != pad_id).to(device)
        unmasked_length = mask.int().sum()
        logits = self(x).permute(0, 2, 1)
        loss = self.loss_fn(logits, y) * mask
        return loss.sum() / unmasked_length
    
    def accuracy(self, x, y):
        unmasked_length = (y != pad_id).int().sum().to(device)
        logits = self(x)
        total_correct = (logits.argmax(-1).view(-1) == y.view(-1)).sum()
        return total_correct / unmasked_length
    
    def sample(self, start_token_id, max_new_tokens, temp=1.0, topk=20):
        ctx = torch.ones((1, 1), dtype=torch.int32).to(device) * start_token_id
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -ctx_len:])[:, -1, :]
            topk_logits, topk_idxs = torch.topk(logits, topk, dim=-1)
            probas = torch.softmax(topk_logits / temp, -1)
            next_tok = topk_idxs[0, torch.multinomial(probas, 1).item()]
            if next_tok == 0:
                return ctx
            ctx = torch.concat((ctx, next_tok.view(1, 1)), dim=1)
        return ctx

### Training Loop
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    lr = 0
    lr_factor = 1
    step = 1
    warmup_steps = 4000
    weight_decay = 0.01
    grad_clip = 1.0

    # Ready validation data
    valid_data = valid_data.to(device)
    valid_ds = TensorDataset(valid_data[:, :-1], valid_data[:, 1:])
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # Define model
    model = MusicTransformer()
    param_groups = model.parameters()

    # Weight Decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {'params': no_decay_params, 'weight_decay': 0.0},  # No weight decay for LayerNorm
        {'params': decay_params, 'weight_decay': weight_decay}  # Weight decay for other parameters
    ]

    # Define optimizer and compile model for efficiency boost
    optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-9)
    model = model.to(device)
    model = torch.compile(model)
    torch.set_float32_matmul_precision("high")

    # wandb for logging
    wandb.login(key="your-api-key")
    run = wandb.init(
        project="project-name",
        name="music-transformer",
        config = {
            "n_emb": n_emb,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "dropout": dropout,
            "head_size": head_size,
            "filter_size": filter_size,
            "ctx_len": ctx_len,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "warmup_steps": warmup_steps,
        }
    )

    # Calculate loss and accuracy on evaluation dataset
    def run_eval():
        model.eval()
        running_loss = 0
        running_acc = 0
        num_val_batches = len(valid_loader)
        for input, label in valid_loader:
            loss = model.loss(input, label)
            running_acc += model.accuracy(input, label)
            running_loss += loss.item()
        avg_valid_loss = running_loss / num_val_batches
        avg_valid_acc = running_acc / num_val_batches
        return avg_valid_loss, avg_valid_acc

    steps = int(1e6)
    eval_steps = 1000
    best_val_loss = 1e9
    for step in range(steps):
        # learning rate schedule
        if step > 0:
            lr = lr_factor * (n_emb ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
            for p in optimizer.param_groups:
                p['lr'] = lr

        if (step % eval_steps == 0):
            loss, acc = run_eval()
            # save best model
            if loss < best_val_loss:
                best_val_loss = loss
                torch.save({
                    'lr': lr,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 
                    f"checkpoint_best.pt"
                )
            wandb.log({"val/loss": loss, "val/accuracy": acc, "step": step})
            # save model checkpoint
            torch.save({
                'lr': lr,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 
                f"checkpoint.pt"
            )

        # training step
        batch = get_batch(batch_size)
        input = batch[:, :-1]
        label = batch[:, 1:] # labels are inputs shifted left
        optimizer.zero_grad()
        loss = model.loss(input, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        wandb.log({"train/loss": loss.item(), "lr": lr, "step": step})