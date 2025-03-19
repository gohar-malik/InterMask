import os
from argparse import Namespace
import re
from os.path import join as pjoin
from utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, complete=True, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip('\n').split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt.device = device

    if complete:
        opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
        opt.model_dir = pjoin(opt.save_root, 'model')
        opt.meta_dir = pjoin(opt.save_root, 'meta')
        opt.anim_dir = pjoin(opt.save_root, 'animation')
        opt.eval_dir = pjoin(opt.save_root, 'eval')
        opt.log_dir = pjoin(opt.save_root, 'log')

        if opt.dataset_name == 'interhuman':
            opt.data_root = 'data/InterHuman'
            opt.joints_num = 22
        elif opt.dataset_name == "interx":
            opt.data_root = 'data/InterX'
            opt.motion_dir = pjoin(opt.data_root, 'motions')
            opt.text_dir = pjoin(opt.data_root, 'texts_processed')
            opt.joints_num = 56
            opt.max_motion_length = 150
        else:
            raise KeyError('Dataset not recognized')
        
        opt.is_train = False
        opt.is_continue = False
    

    opt_dict.update(kwargs) # Overwrite with kwargs params

    return opt