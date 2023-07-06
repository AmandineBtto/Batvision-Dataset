
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
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")  
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    #print(OmegaConf.to_yaml(cfg))
    
    if cfg.mode.mode != 'train':
        raise Exception('This script is for training only. Please run test.py for evaluation')
    if cfg.model.name != 'unet_baseline':
        raise Exception('This script if for training on unet model only')

    # ------------ GPU config ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    print("{} {} device is used".format(n_GPU,device))

    batch_size = cfg.mode.batch_size
    
    # ------------ Create dataset -----------
        
    # Use corresponding dataset class
    if cfg.dataset.name == 'batvisionv1':
        train_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train)
        if cfg.mode.validation:
            val_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_val)
    elif cfg.dataset.name == 'batvisionv2':
        train_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train) 
        if cfg.mode.validation:
            val_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val) 
    else:
        raise Exception('Training can be done only on BV1 and BV2')

    print(f'Train Dataset of {len(train_set)} instances')
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=cfg.mode.shuffle, num_workers=cfg.mode.num_threads) 

    if cfg.mode.validation:
        print(f'Validation Dataset of {len(val_set)} instances')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=cfg.mode.shuffle, num_workers=cfg.mode.num_threads)


    # ---------- Load Model ----------
    model = define_G(cfg, input_nc = 2, output_nc = 1, ngf = 64, netG = cfg.model.generator, norm = 'batch',
                                    use_dropout = False, init_type='normal', init_gain=0.02, gpu_ids = [device])
    print('Model used:', cfg.model.generator)
   
    # ---------- Criterion & Optimizers ----------
    if cfg.mode.criterion == 'L1':
        criterion = nn.L1Loss().to(device)
    
    learning_rate = cfg.mode.learning_rate

    if cfg.mode.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # ---------- TensorBoard ---------- 
    
    experiment_name = cfg.model.generator + '_' +  cfg.dataset.name + '_' + 'BS' + str(cfg.mode.batch_size) + '_' + 'Lr' + str(cfg.mode.learning_rate) + '_' + cfg.mode.optimizer + '_' + cfg.mode.experiment_name

    writer = SummaryWriter('./logs/' + experiment_name + '/')
    file = open("./logs/"+ experiment_name +'/'+"architecture.txt","w")

    # dataloader parameters
    file.write("Dataset name: {}\n".format(cfg.dataset.name))
    file.write("Batch size: {}\n".format(batch_size))
    file.write("Image processing: {}\n".format(cfg.dataset.preprocess))
    file.write("Image resize: {}\n".format(cfg.dataset.images_size))
    file.write("Depth norm: {}\n".format(cfg.dataset.depth_norm))
    file.write("Audio type used for training: {}\n".format(cfg.dataset.audio_format))
    
    # parameters
    file.write("Learning rate: {}\n".format(cfg.mode.learning_rate))
    file.write("Optimize used : {}\n".format(cfg.mode.optimizer))

    # net architecture
    file.write("Generator: {}\n".format(cfg.model.generator))
    
    file.write(str(model))
    file.close()


    if cfg.mode.checkpoints is None:
        checkpoint_epoch=1
    else:
        load_epoch = cfg.mode.checkpoints
        checkpoint = torch.load('./checkpoints/' + experiment_name + '/checkpoint_' + str(load_epoch) + '.pth')
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_epoch = checkpoint["epoch"] + 1 

    nb_epochs = cfg.mode.epochs

    for param_group in optimizer.param_groups:
        print("Learning rate used: {}".format(param_group['lr']))

    train_iter = 0
    for epoch in range(checkpoint_epoch, nb_epochs + 1):

        t0 = time.time()

        batch_loss = [] 
        batch_loss_val = [] 

        # ------ Training ---------
        model.train()  

        for i,(audio, gtdepth) in enumerate(train_loader):

            audio = audio.to(device)
            gtdepth = gtdepth.to(device)   

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            depth_pred = model(audio)
            
            # compute loss
            loss = criterion(depth_pred[gtdepth!=0], gtdepth[gtdepth!=0]) 
            batch_loss.append(loss.item()) 
            
            # optimize
            loss.backward()  # backward-pass
            optimizer.step()  # update weights
            
            train_iter +=1
            
        writer.add_scalar('Train/Loss', np.mean(batch_loss), epoch) 
    
        epoch_time = time.time()-t0
        print(' - epoch time: {:.1f}'.format(epoch_time))

        writer.add_scalar('Epoch_time',epoch_time,epoch)

        # ------- Validation ------------
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            model.eval() 
            errors = []
            with torch.no_grad():
                for audio_val, gtdepth_val in val_loader:
                    audio_val = audio_val.to(device)
                    gtdepth_val = gtdepth_val.to(device)        

                    depth_pred_val = model(audio_val)
                    
                    loss_val = criterion(depth_pred_val[gtdepth_val!=0],gtdepth_val[gtdepth_val!=0])
                    batch_loss_val.append(loss_val.item()) 

                    for idx in range(depth_pred_val.shape[0]):
                        if cfg.dataset.depth_norm:
                            # if normalization, return to true range for metrics computation
                            errors.append(compute_errors(gtdepth_val[idx].cpu().numpy() * cfg.dataset.max_depth, 
                                    depth_pred_val[idx].cpu().numpy() * cfg.dataset.max_depth))
                        else:
                            errors.append(compute_errors(gtdepth_val[idx].cpu().numpy(), 
                                    depth_pred_val[idx].cpu().numpy()))
	
                mean_errors = np.array(errors).mean(0)	
                print('RMSE: {:.3f}'.format(mean_errors[1])) 
                val_errors = {}
                val_errors['ABS_REL'], val_errors['RMSE'] = mean_errors[0], mean_errors[1]
                val_errors['DELTA1'] = mean_errors[2] 
                val_errors['DELTA2'] = mean_errors[3]
                val_errors['DELTA3'] = mean_errors[4]

                writer.add_scalar('Val/Loss', np.mean(batch_loss_val), epoch)
                writer.add_scalar('Val/RMSE', val_errors['RMSE'], epoch)
                writer.add_scalar('Val/ABS_REL', val_errors['ABS_REL'], epoch)
                writer.add_scalar('Val/DELTA1', val_errors['DELTA1'], epoch)
                writer.add_scalar('Val/DELTA2', val_errors['DELTA2'], epoch)
                writer.add_scalar('Val/DELTA3', val_errors['DELTA3'], epoch)

                if epoch % cfg.mode.validation_display == 0:
                    tensorboard_display_input_pred(writer, audio_val, depth_pred_val, gtdepth_val, 'Val', epoch)

        # ------- Display tensorboard ------------
        if epoch % cfg.mode.print_tensorboard == 0:
            tensorboard_display_input_pred(writer, audio, depth_pred, gtdepth, 'Train', epoch)

        # ------- Save ------------
        if epoch % cfg.mode.saving_checkpoints == 0:
            print('Save network')
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            path_check = './checkpoints/' + experiment_name + '/'
            isExist = os.path.exists(path_check)
            if not isExist:
                os.makedirs(path_check)
            torch.save(state, './checkpoints/' + experiment_name + '/checkpoint_' + str(epoch) + '.pth')

    
if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("Exception happened during training")