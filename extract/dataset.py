import os
import pickle

DATASETS = ['roxford5k', 'rparis6k']

def configdataset(dataset, dir_main):

    dataset = dataset.lower()

    if dataset not in DATASETS:    
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    # loading imlist, qimlist, and gnd, in cfg as a dict
    gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
    cfg['gnd_fname'] = gnd_fname
    
    cfg['ext'] = '.jpg'
    cfg['qext'] = '.jpg'
    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])
 
    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset
    
    images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
    
    return cfg

def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])
