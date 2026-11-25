from graph import GR, Margin, diff_prior, ref_bank, np, torch, nn
from prepare_stage import CONFIG, TASK, training_inputs
from threadpoolctl import threadpool_limits

OUT = TASK / 'outputs/weights'

def train(x, a, y, pid, fold):
    OUT.mkdir(exist_ok=True)

    for k in range(CONFIG['folds']):
        train=np.flatnonzero(fold!=k);test=np.flatnonzero(fold==k)
        assert not set(pid[train])&set(pid[test])
        rng=np.random.default_rng(9042+k);torch.manual_seed(62850+k)
        model=Margin(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.0013,weight_decay=.045)
        ref_bank(x,a,y,pid,train,rng)  # Advance the historical reference-sampling RNG.

        _,ii,cnt=np.unique(pid[train],return_inverse=True,return_counts=True)
        w=1/cnt[ii];w/=np.bincount(y[train],weights=w)[y[train]];w/=w.sum()
        grade=np.rint(a@GR).clip(0,9).astype(int)
        pool={g:train[abs(grade[train]-g)<=1] for g in range(10)}
        for e in range(1, CONFIG['epoch'] + 1):
            model.train()
            left=rng.choice(train,36000,replace=True,p=w)
            right=rng.choice(train,len(left),replace=True,p=w)
            nearby=rng.random(len(left))<.64
            for g in range(10):
                keep=nearby&(grade[left]==g);right[keep]=rng.choice(pool[g],keep.sum())
            good=pid[left]!=pid[right];left=left[good];right=right[good]
            for off in range(0,len(left),400):
                i=left[off:off+400];j=right[off:off+400]
                prior=np.log(diff_prior(a[i],a[j])).astype('f');label=torch.tensor(y[i]-y[j]+9,dtype=torch.long)
                dd=model(torch.from_numpy(x[i]),torch.from_numpy(x[j]))
                loss=nn.functional.cross_entropy(torch.from_numpy(prior)+dd,label)+.022*dd.square().mean()
                opt.zero_grad();loss.backward();opt.step()
        torch.save(model.state_dict(), OUT / f'fold{k}.pt')
        print('independent reference fold',k,len(train),flush=True)


def main():
    torch.set_num_threads(2)
    with threadpool_limits(2):
        train(*training_inputs())
