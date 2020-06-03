from easydict import EasyDict as edict

task_to_out_nc = edict(autoencoder=3, depths=1, normals=3,)


def get_acc_at(lnorm_dist_map, val):
    acc = (lnorm_dist_map <= val).flatten(1).sum(1).float() / lnorm_dist_map.size(1)
    return acc
