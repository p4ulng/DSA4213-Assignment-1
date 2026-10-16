import argparse
import json
import math
from collections import Counter
from pathlib import Path

class InterpolatedTrigram:
    def __init__(self,vocab,weights=(0.1,0.3,0.6),alpha=0.1):
        # Blend unigram, bigram, and trigram estimates; alpha smooths counts.
        self.vocab=vocab; self.weights=weights; self.alpha=alpha # weights are the lambdas for interpolation and alpha is the "k" for k-smoothing
    def fit(self,ids):
        # Count each token, adjacent pair, and adjacent triple in the training stream.
        self.uni=Counter(ids); self.bi=Counter(zip(ids[:-1],ids[1:])); self.tri=Counter(zip(ids[:-2],ids[1:-1],ids[2:])); self.total=len(ids)
    def prob(self,a,b,c):
        # Estimate next-token c from no context, one-token context b, and two-token context a,b.
        v=self.vocab; s=self.alpha
        p1=(self.uni[c]+s)/(self.total+s*v)
        p2=(self.bi[b,c]+s)/(self.uni[b]+s*v)
        p3=(self.tri[a,b,c]+s)/(self.bi[a,b]+s*v)
        return self.weights[0]*p1+self.weights[1]*p2+self.weights[2]*p3
    def evaluate(self,ids):
        # Score every token from position 2 onward and convert mean negative log probability to perplexity.
        return math.exp(-sum(math.log(self.prob(ids[i-2],ids[i-1],ids[i])) for i in range(2,len(ids)))/(len(ids)-2))

def main():
    # Command-line baseline: fit on training IDs, then report held-out perplexity.
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/processed_3m_max_tokens_30k_vocab'); a=p.parse_args(); root=Path(a.data)
    load=lambda s:json.loads((root/f'{s}.json').read_text())
    vocab=json.loads((root/'vocab.json').read_text())
    model=InterpolatedTrigram(len(vocab)); model.fit(load('train'))
    print(json.dumps({'model':'interpolated_trigram','validation_ppl':model.evaluate(load('validation')),'test_ppl':model.evaluate(load('test'))},indent=2))
if __name__=='__main__':main()
