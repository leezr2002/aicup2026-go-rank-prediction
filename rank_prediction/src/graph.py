import numpy as np
import torch
from torch import nn
from scipy.special import softmax

GRADE = GR = np.arange(10)

def logp(p):
    return np.log(np.asarray(p, dtype='d').clip(1e-8))

def sample_input(base,tree,fp,cc,kn,hmm,n):
    anchor=softmax(logp(tree)+.55*(logp(fp)-logp(base)).clip(-4,4),1)
    curves=[];stats=[]
    for p in (anchor,base,fp,cc,kn,hmm):
        lp=logp(p);mu=p@GR
        curves.extend([np.sqrt(p),np.diff(lp,axis=1).clip(-5,5)/4])
        stats.extend([mu/3-1.5,np.sqrt((p@(GR**2)-mu*mu).clip(0,16)),(-p*lp).sum(1)])
    x=np.column_stack([*curves,*stats,np.log2(np.maximum(n,2)/8)])
    return x.astype('f'),anchor.astype('f')

class Margin(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(d,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.12),nn.Linear(48,24))
        self.head=nn.Sequential(nn.Linear(24*4,56),nn.LayerNorm(56),nn.SiLU(),nn.Dropout(.18),nn.Linear(56,19))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,y):
        a,b=self.encoder(x),self.encoder(y)
        return 2.3*torch.tanh(self.head(torch.cat([a,b,a-b,a*b],dim=1))/2.3)

def diff_prior(a,b):
    # Prior likelihood of an ordinal difference between independent uncertain grades.
    raw=a[:,:,None]*b[:,None,:]
    # Axis ordering is explicit: d = row grade (query) minus column grade (reference).
    return np.column_stack([np.diagonal(raw,offset=-d,axis1=1,axis2=2).sum(1) for d in range(-9,10)]).clip(1e-8)

def ref_bank(x,a,y,pid,indices,rng):
    keep=[]
    for k in range(10):
        rows=indices[y[indices]==k];keys,choose=np.unique(pid[rng.permutation(rows)],return_index=True)
        keys=rng.permutation(keys)[:110]
        for name in keys:
            possible=rows[pid[rows]==name];keep.append(rng.choice(possible))
    keep=np.array(keep)
    return x[keep],a[keep],y[keep]

def edges(x,anchor):
    # A local comparison needs overlapping grade ranges. Similar profile disagreement and
    # uncertainty help select comparable players within that range.
    order=anchor@GRADE
    dr=np.abs(order[:,None]-order)
    p=np.sqrt(anchor);hell=np.maximum(0,1-p@p.T)
    sk=x[:,-19:-1];sd=sk.std(0).clip(.2);a=sk/sd
    sq=np.maximum(0,(a*a).mean(1)[:,None]+(a*a).mean(1)-2*(a@a.T)/a.shape[1])
    dist=dr+.8*hell+.13*sq+.09*np.abs(x[:,-1,None]-x[:,-1])
    np.fill_diagonal(dist,np.inf)
    count=min(12,len(x)-1);sel=np.argsort(dist,axis=1,kind='stable')[:,:count]
    return sel,np.take_along_axis(dist,sel,axis=1)

def factors(x,sel,models):
    i=np.repeat(np.arange(len(x)),sel.shape[1]);j=sel.ravel()
    res=np.zeros((len(i),19),'f')
    with torch.no_grad():
        for m in models:
            enc=m.encoder(torch.from_numpy(x))
            for off in range(0,len(i),2000):
                dest=slice(off,off+2000);a=enc[i[dest]];b=enc[j[dest]]
                def raw(p,q):
                    return 2.3*torch.tanh(m.head(torch.cat([p,q,p-q,p*q],1))/2.3)
                res[dest]+=(.5*(raw(a,b)+raw(b,a).flip(1))).numpy()/len(models)
    return res.reshape(len(x),sel.shape[1],19)

def message(ref,sel,dist,f,n,rounds=2):
    # Temper messages by the effective number of neighbors to avoid counting correlated evidence repeatedly.
    sel=sel[:,:n];dist=dist[:,:n];f=f[:,:n]
    w=np.exp(-dist);w/=w.sum(1,keepdims=True).clip(1e-9)
    idx=GRADE[:,None]-GRADE[None,:]+9;pair=np.exp(f[:,:,idx].clip(-2.3,2.3))
    current=ref
    for step in range(rounds):
        marginal=np.einsum('nijk,nik->nij',pair,current[sel]).clip(1e-8)
        pooled=(w[:,:,None]*np.log(marginal)).sum(1)
        pooled-=np.sum(pooled*ref,1,keepdims=True)
        current=softmax(logp(ref)+.48*pooled,1)
    return pooled.clip(-2,2)
