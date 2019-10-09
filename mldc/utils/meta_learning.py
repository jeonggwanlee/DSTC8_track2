import torch

def divide_in_and_out_batchsz(self, spt_or_que_set: torch.Tensor, batchsz=1, mode='support', mc_num=2, max_len=300, manual_size=-1):
    """# mode 'support' or 'meta-train-query' """

    ii, mti, ll, ml, tti = spt_or_que_set
    orig_batchsz = ii.shape[0]
    if manual_size == -1:
        manual_size = (orig_batchsz // batchsz) * batchsz
    perm = torch.randperm(orig_batchsz)[:manual_size]
    spt_or_que_set = tuple([gpt_inp_elem[perm] for gpt_inp_elem in spt_or_que_set])
    ii, mti, ll, ml, tti = spt_or_que_set

    if mode == 'meta-train-query':
        ii = ii[:, 1, :].reshape([ii.shape[0], -1, ii.shape[2]])
        mti = mti[:, 1].reshape([mti.shape[0], -1])
        ll = ll[:, 1, :].reshape([ll.shape[0], -1, ll.shape[2]])
        tti = tti[:, 1, :].reshape([tti.shape[0], -1, tti.shape[2]])
        spt_or_que_set = (ii, mti, ll, ml, tti)
        assert (mc_num == 1)
    elif mode == 'support':
        assert (mc_num == 2)

    ii = ii.reshape([-1, batchsz, mc_num, max_len])
    mti = mti.reshape([-1, batchsz, mc_num])
    ll = ll.reshape([-1, batchsz, mc_num, max_len])
    ml = ml.reshape([-1, batchsz])
    tti = tti.reshape([-1, batchsz, mc_num, max_len])

    if mode == 'support':
        spt_or_que_set = (ii, mti, ll, ml, tti)
    elif mode == 'meta-train-query':
        spt_or_que_set = (ii, [None] * ii.shape[0], ll, [None] * ii.shape[0], tti)

    return spt_or_que_set