import torch

def calc_cdist(feat1, feat2, batch_size=10000):
    dists = []
    for feat2_batch in feat2.split(batch_size):
        dists.append(torch.cdist(feat1, feat2_batch).cpu())
    return torch.cat(dists, dim=1)


def calculate_precision_recall_part(feat_r, feat_g, k=3, batch_size=10000):
    # Precision
    NNk_r = []
    for feat_r_batch in feat_r.split(batch_size):
        NNk_r.append(calc_cdist(feat_r_batch, feat_r, batch_size).kthvalue(k+1).values)
    NNk_r = torch.cat(NNk_r)
    precision = []
    for feat_g_batch in feat_g.split(batch_size):
        dist_g_r_batch = calc_cdist(feat_g_batch, feat_r, batch_size)
        precision.append((dist_g_r_batch <= NNk_r).any(dim=1).float())
    precision = torch.cat(precision).mean().item()
    # Recall
    NNk_g = []
    for feat_g_batch in feat_g.split(batch_size):
        NNk_g.append(calc_cdist(feat_g_batch, feat_g, batch_size).kthvalue(k+1).values)
    NNk_g = torch.cat(NNk_g)
    recall = []
    for feat_r_batch in feat_r.split(batch_size):
        dist_r_g_batch = calc_cdist(feat_r_batch, feat_g, batch_size)
        recall.append((dist_r_g_batch <= NNk_g).any(dim=1).float())
    recall = torch.cat(recall).mean().item()
    return precision, recall


def calc_cdist_full(feat1, feat2, batch_size=10000):
    dists = []
    for feat1_batch in feat1.split(batch_size):
        dists_batch = []
        for feat2_batch in feat2.split(batch_size):
            dists_batch.append(torch.cdist(feat1_batch, feat2_batch).cpu())
        dists.append(torch.cat(dists_batch, dim=1))
    return torch.cat(dists, dim=0)


def calculate_precision_recall_full(feat_r, feat_g, k=3, batch_size=10000):
    NNk_r = calc_cdist_full(feat_r, feat_r, batch_size).kthvalue(k+1).values
    NNk_g = calc_cdist_full(feat_g, feat_g, batch_size).kthvalue(k+1).values
    dist_g_r = calc_cdist_full(feat_g, feat_r, batch_size)
    dist_r_g = dist_g_r.T
    # Precision
    precision = (dist_g_r <= NNk_r).any(dim=1).float().mean().item()
    # Recall
    recall = (dist_r_g <= NNk_g).any(dim=1).float().mean().item()
    return precision, recall


def calculate_precision_recall(feat_r, feat_g, device=torch.device('cuda'), k=3,
                               batch_size=10000, save_cpu_ram=False, **kwargs):
    feat_r = feat_r.to(device)
    feat_g = feat_g.to(device)
    if save_cpu_ram:
        return calculate_precision_recall_part(feat_r, feat_g, k, batch_size)
    else:
        return calculate_precision_recall_full(feat_r, feat_g, k, batch_size)

