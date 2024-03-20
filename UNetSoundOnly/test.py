
from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset

from models.utils_models import *

from models.unetbaseline_model import *

from utils_tensorboard import *
from utils_criterion import compute_errors

import time
import os 
import numpy as np 
import math
import pickle
import pandas as pd

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")  
def main(cfg):

    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    
    if cfg.mode.mode != 'test':
        raise Exception('This script is for test only. Please run train.py for training')

    # ------------ GPU config ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    print("{} {} device is used".format(n_GPU,device))

    batch_size = cfg.mode.batch_size
    
    # ------------ Create dataset -----------
        
    # Use corresponding dataset class
    if cfg.dataset.name == 'batvisionv1':
        if cfg.mode.eval_on == 'val':
            eval_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_val)
        else:
            eval_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_test)
    elif cfg.dataset.name == 'batvisionv2':
        if cfg.mode.eval_on == 'val':
            eval_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val) 
        else:
            eval_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_test) 
    else:
        raise Exception('Test can be done only on BV1 and BV2')

    print(f'Eval Dataset of {len(eval_set)} instances')
    eval_loader = DataLoader(eval_set, batch_size = batch_size, shuffle=False, num_workers=cfg.mode.num_threads) 

    # ---------- Load Model ----------
    
 
    model = define_G(cfg, input_nc = 2, output_nc = 1, ngf = 64, netG = cfg.model.generator, norm = 'batch',
                                    use_dropout = False, init_type='normal', init_gain=0.02, gpu_ids = [device])
    print('Network used:', cfg.model.generator)
    
    if cfg.mode.criterion == 'L1':
        criterion = nn.L1Loss().to(device)
    
    if cfg.mode.checkpoints is None:
            raise AttributeError('In test mode, a checkpoint needs to be loaded.')
    else:
        load_epoch = cfg.mode.checkpoints
        checkpoint = torch.load('./checkpoints/' + cfg.mode.experiment_name + '/checkpoint_' + str(load_epoch) + '.pth')
        model.load_state_dict(checkpoint["state_dict"])
        print('Epoch loaded:', str(load_epoch))
    

    # ------ Eval ---------
    model.eval()  # eval mode

    gt_imgs_to_save = []
    pred_imgs_to_save = []
    loss_list = []
    errors = []
    rmse_list = []
    abs_rel_list = []
    log10_list = []
    delta1_list = []
    delta2_list = []
    delta3_list = [] 
    mae_list = []

    with torch.no_grad():

        for audio, depthgt in eval_loader:

            audio = audio.to(device)
            depthgt = depthgt.to(device)        

            depth_pred = model(audio)

            loss_test = criterion(depth_pred[depthgt !=0], depthgt[depthgt !=0]) 
            loss_list.append(loss_test.cpu().item())

            for idx in range(depth_pred.shape[0]):
                gt_imgs_to_save.append(depthgt[idx].detach().cpu().numpy())
                pred_imgs_to_save.append(depth_pred[idx].detach().cpu().numpy())
                if cfg.dataset.depth_norm:
                    unscaledgt = depthgt[idx].detach().cpu().numpy() * cfg.dataset.max_depth
                    unscaledpred = depth_pred[idx].detach().cpu().numpy() * cfg.dataset.max_depth
                    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(unscaledgt, 
                        unscaledpred)
                else:   
                    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(depthgt[idx].cpu().numpy(), 
                            depth_pred[idx].cpu().numpy())
                errors.append((abs_rel, rmse, a1, a2, a3, log_10, mae))
            
            rmse_list.append(rmse)
            abs_rel_list.append(abs_rel)
            log10_list.append(log_10)
            delta1_list.append(a1)
            delta2_list.append(a2)
            delta3_list.append(a3)
            mae_list.append(mae)


        mean_errors = np.array(errors).mean(0)	
        print('abs rel: {:.3f}'.format(mean_errors[0])) 
        print('RMSE: {:.3f}'.format(mean_errors[1])) 
        print('Delta1: {:.3f}'.format(mean_errors[2])) 
        print('Delta2: {:.3f}'.format(mean_errors[3])) 
        print('Delta3: {:.3f}'.format(mean_errors[4])) 
        print('Log10: {:.3f}'.format(mean_errors[5])) 
        print('MAE: {:.3f}'.format(mean_errors[6])) 
    
    # Save evaluation
    d = {'loss': loss_list, 'abs_rel': abs_rel_list, 'rmse': rmse_list, 'log10': log10_list, 'delta1': delta1_list, 'delta2': delta2_list, 'delta3': delta3_list, 'mae': mae_list, 'gt_images': gt_imgs_to_save, 'pred_imgs': pred_imgs_to_save}

    stats_df = pd.DataFrame(data=d)
    if cfg.mode.eval_on == 'test':
        stats_df.to_pickle(os.path.join(cfg.mode.stat_dir + cfg.dataset.name, 'test',  'stats_on_' +  cfg.dataset.name + '_test_set_'+ cfg.mode.experiment_name + '_epoch_' + str(load_epoch) +".pkl"))
    else:
        stats_df.to_pickle(os.path.join(cfg.mode.stat_dir + cfg.dataset.name, 'val', 'stats_on_' + cfg.dataset.name + '_val_set_'+ cfg.mode.experiment_name + '_epoch_' + str(load_epoch) + ".pkl")) 



if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("Exception happened during test")
