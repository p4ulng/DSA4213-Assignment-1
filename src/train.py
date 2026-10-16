import argparse
import csv
import json
import math
import random
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import build_model

def get_device(choice):
    # Honor an explicit device; auto prefers Apple's MPS backend when available.
    if choice!='auto':return torch.device(choice)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def hardware_metadata(device):
    meta={'system':platform.platform(),'architecture':platform.machine(),'processor':platform.processor(),'torch_version':torch.__version__,'device':str(device)}
    if platform.system()=='Darwin':
        for key,command in [('model',['sysctl','-n','hw.model']),('chip',['sysctl','-n','machdep.cpu.brand_string']),('memory_bytes',['sysctl','-n','hw.memsize'])]:
            try:
                value=subprocess.check_output(command,text=True,stderr=subprocess.DEVNULL).strip()
                meta[key]=int(value) if key=='memory_bytes' else value
            except (OSError,subprocess.CalledProcessError,ValueError):
                pass
    return meta

def synchronize(device):
    if device.type=='mps': torch.mps.synchronize()

def batches(ids,context,batch_size,shuffle=False,seed=42):
    # Non-overlapping target windows: every token except the first is scored once.
    # Start fixed-width chunks; within a chunk, x[t] predicts the following token y[t].
    starts=list(range(0,len(ids)-1,context))
    if shuffle: random.Random(seed).shuffle(starts)
    for j in range(0,len(starts),batch_size):
        chunk=starts[j:j+batch_size]
        x=torch.zeros(len(chunk),context,dtype=torch.long)
        # -100 is PyTorch's ignored target ID, so padded tail positions add no loss.
        y=torch.full((len(chunk),context),-100,dtype=torch.long)
        for row,s in enumerate(chunk):
            n=min(context,len(ids)-s-1)
            x[row,:n]=torch.tensor(ids[s:s+n]); y[row,:n]=torch.tensor(ids[s+1:s+n+1])
        yield x,y

def epoch(model,ids,context,batch_size,device,opt=None,max_batches=0,seed=42):
    # Passing an optimizer means training; without one, this is validation/test evaluation.
    training=opt is not None; model.train(training)
    total_loss=0.; total_tokens=0
    iterator=batches(ids,context,batch_size,shuffle=training,seed=seed)
    for step,(x,y) in enumerate(iterator):
        if max_batches and step>=max_batches:break
        x=x.to(device); y=y.to(device)
        with torch.set_grad_enabled(training):
            logits=model(x)
            # Flatten batch/time dimensions; ignore padded target slots in cross-entropy.
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100)
            if training:
                # Standard gradient update; clipping limits unusually large recurrent gradients.
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        n=(y!=-100).sum().item(); total_loss+=loss.item()*n; total_tokens+=n
    return total_loss/max(total_tokens,1),total_tokens

def main():
    # CLI arguments define the experiment so runs can be repeated from the notebook or terminal.
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['rnn','lstm','transformer'],required=True)
    p.add_argument('--data',default='data/processed_3m_max_tokens_30k_vocab'); p.add_argument('--out',default='runs')
    p.add_argument('--epochs',type=int,default=25); p.add_argument('--patience',type=int,default=5,help='stop after this many consecutive epochs without validation-loss improvement'); p.add_argument('--context',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--emb',type=int,default=100)
    p.add_argument('--hidden',type=int,default=128); p.add_argument('--heads',type=int,default=4); p.add_argument('--layers',type=int,default=2)
    p.add_argument('--embedding-init',default='',help='optional .npy matrix aligned to vocab.json')
    p.add_argument('--freeze-embedding',action='store_true',help='keep the supplied/input embedding table fixed')
    p.add_argument('--lr',type=float,default=0.001); p.add_argument('--device',default='auto')
    p.add_argument('--max-batches',type=int,default=0,help='pilot only: limits batches in each epoch')
    p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    # Seed Python, NumPy, and PyTorch randomness for reproducibility.
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    root=Path(a.data); load=lambda s:json.loads((root/f'{s}.json').read_text())
    vocab=json.loads((root/'vocab.json').read_text()); train=load('train'); valid=load('validation'); test=load('test')
    if a.epochs < 1:
        p.error('--epochs must be at least 1')
    if a.patience < 1:
        p.error('--patience must be at least 1')
    device=get_device(a.device)
    embedding_weights=None
    if a.embedding_init:
        embedding_weights=torch.from_numpy(np.load(a.embedding_init)).float()
        if embedding_weights.shape != (len(vocab),a.emb):
            raise ValueError(f'Embedding matrix must have shape {(len(vocab),a.emb)}, got {tuple(embedding_weights.shape)}')
    if a.freeze_embedding and embedding_weights is None:
        raise ValueError('--freeze-embedding requires --embedding-init')
    # With no --embedding-init, nn.Embedding uses random initialization; unless
    # --freeze-embedding is set, the random table is learned jointly with the LM.
    # Part II passes --embedding-init and --freeze-embedding for fixed Skip-gram/GloVe.
    # Build the selected architecture and move its parameters to the chosen device.
    model=build_model(a.model,len(vocab),a.emb,a.hidden,a.heads,a.layers,max(a.context,256),embedding_weights,a.freeze_embedding).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    run=Path(a.out)/a.model; run.mkdir(parents=True,exist_ok=True)
    data_metadata=json.loads((root/'metadata.json').read_text()) if (root/'metadata.json').exists() else {}
    (run/'config.json').write_text(json.dumps(vars(a)|{'resolved_device':str(device),'parameters':sum(p.numel() for p in model.parameters()),'hardware':hardware_metadata(device),'tokenizer':data_metadata.get('tokenizer'),'vocab_size':len(vocab),'train_token_ids':len(train)},indent=2))
    history=[]; best=float('inf'); best_epoch=0; epochs_without_improvement=0; train_seconds=0.0
    # Train one epoch, measure validation loss, and stop after `patience` stale epochs.
    for i in range(a.epochs):
        synchronize(device); t=time.perf_counter(); tl,tn=epoch(model,train,a.context,a.batch_size,device,opt,a.max_batches,a.seed+i); synchronize(device)
        train_seconds += time.perf_counter()-t
        epoch_seconds=time.perf_counter()-t
        vl,vn=epoch(model,valid,a.context,a.batch_size,device,max_batches=a.max_batches)
        if vl<best:
            best=vl; best_epoch=i+1; epochs_without_improvement=0
            torch.save(model.state_dict(),run/'best.pt') # save best point
        else:
            epochs_without_improvement+=1
        row={'epoch':i+1,'train_loss':tl,'val_loss':vl,'val_ppl':math.exp(min(vl,100)),'train_seconds':round(epoch_seconds,2),'seconds':round(time.perf_counter()-t,2),'train_tokens':tn,'val_tokens':vn,'best_epoch':best_epoch,'epochs_without_improvement':epochs_without_improvement}
        history.append(row); print(json.dumps(row),flush=True)
        if epochs_without_improvement>=a.patience:
            print(f'Early stopping at epoch {i+1}: validation loss did not improve for {a.patience} consecutive epochs (best epoch {best_epoch}, val_loss={best:.6f}).',flush=True)
            break
    with (run/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    # Report test performance for the best-validation epoch, not necessarily the last epoch.
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True))
    loss,tokens=epoch(model,test,a.context,a.batch_size,device,max_batches=a.max_batches)
    result={'model':a.model,'test_loss':loss,'test_ppl':math.exp(min(loss,100)),'test_tokens':tokens,
            'epochs_completed':len(history),'epoch_limit':a.epochs,'patience':a.patience,'best_epoch':best_epoch,'early_stopped':len(history)<a.epochs,
            'training_seconds':round(train_seconds,2),
            'training_time_scope':'sum of synchronized training epochs; excludes validation, startup, and checkpoint I/O',
            'pilot_only':bool(a.max_batches)}
    (run/'results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
