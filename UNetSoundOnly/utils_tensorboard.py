import os 
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Pred and Ground Truth visualization with tensorboard
def plot_specgram(batch_spec_tensor, label):
    '''
    Create a figure containing the spectrograms of a batch.
    '''
    nb_display = batch_spec_tensor.shape[0]
    figure, axes = plt.subplots(1, nb_display, figsize = (10, 10))

    for i in range(0, nb_display):
        spec = axes[i].imshow(batch_spec_tensor[i,...].cpu().float().numpy(), origin = 'lower', cmap = 'magma')
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(spec, cax=cax)
    
    plt.tight_layout()
    plt.ylabel(label)
    return figure

def plot_depth(batch_depth_tensor, label):
    '''
    Create a figure containing the depth of a batch.
    '''
    nb_display = batch_depth_tensor.shape[0]
    figure, axes = plt.subplots(1, nb_display, figsize = (10, 10))

    for i in range(0, nb_display):
        depth = axes[i].imshow(batch_depth_tensor[i,0,...].cpu().float().numpy(), cmap = 'jet')
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(depth, cax=cax)
    
    plt.tight_layout()
    plt.ylabel(label)
    return figure

def tensorboard_display_img(writer, input, pred, gt, mode, epoch):
    '''
    Use plot_specram and plot_depth to write the different images in tensorboard.
    '''
    input_l_img = plot_specgram(input[:4,0,:,:].data, 'input ch0 spec')
    input_r_img = plot_specgram(input[:4,1,:,:].data, 'input ch1 spec')
    writer.add_figure(mode + '/' + 'input ch0 spec', input_l_img, epoch)
    writer.add_figure(mode + '/' + 'input chA spec', input_r_img, epoch)

    pred_depth = plot_depth(pred[:4,:,:,:].data, 'predicted depth')
    writer.add_figure(mode + '/' +  'pred depth', pred_depth, epoch)
    gt_depth = plot_depth(gt[:4,:,:,:].data, 'gt depth')
    writer.add_figure(mode + '/' +  'gt depth', gt_depth, epoch)

def plot_input(batch_input_tensor, nb_display = 4):
    '''
    nb_display is used to choose the number of elements of the batch to display.
    '''

    figure, axes = plt.subplots(2, nb_display, figsize = (16, 16))
    for i in range(0, nb_display):
        for c in range(0,2):
            spec = axes[c, i].imshow(batch_input_tensor[i,c,...].data.cpu().float().numpy(), origin = 'lower', cmap = 'magma')
            divider = make_axes_locatable(axes[c, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(spec, cax=cax)
            axes[c, i].set_title('input_spec_ch_' + str(c), size = 8)
    plt.tight_layout()
    return figure

def plot_pred_gt(batch_pred_tensor, batch_gt_tensor, nb_display = 4):
    '''
    nb_display is used to choose the number of elements of the batch to display.
    '''

    figure, axes = plt.subplots(2, nb_display, figsize = (16, 16))
    for i in range(0, nb_display):
        pred_depth = axes[0, i].imshow(batch_pred_tensor[i,0,...].data.cpu().float().numpy(), cmap = 'jet')
        divider = make_axes_locatable(axes[0, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pred_depth, cax=cax)
        axes[0,i].set_title('pred', size = 8)

        gt_depth = axes[1, i].imshow(batch_gt_tensor[i,0,...].data.cpu().float().numpy(), cmap = 'jet')
        divider = make_axes_locatable(axes[1, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(gt_depth, cax=cax)
        axes[1,i].set_title('gt', size = 8)

    plt.tight_layout()
    return figure

def tensorboard_display_input_pred(writer, input, pred, gt, mode, epoch):
    input_figure = plot_input(input)
    output_figure = plot_pred_gt(pred, gt)
    writer.add_figure(mode + '/' +  'Input Spec', input_figure, epoch)
    writer.add_figure(mode + '/' +  'Output Depth', output_figure, epoch)
