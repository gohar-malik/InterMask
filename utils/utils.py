import math
import time
import torch
import random
import numpy as np

def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('ep/it:%2d-%4d niter:%6d' % (epoch, inner_iter, niter_state), end=" ")

    message = ' %s completed:%3d%%)' % (time_since(start_time, niter_state / total_niters), niter_state / total_niters * 100)
    # now = time.time()
    # message += '%s'%(as_minutes(now - start_time))


    for k, v in losses.items():
        message += ' %s: %.5f ' % (k, v)
    # message += ' sl_length:%2d tf_ratio:%.2f'%(sl_steps, tf_ratio)
    print(message)

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)