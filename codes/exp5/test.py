import numpy as np 
import torch
import torch.nn as nn
import os, logging, glob
from natsort import natsorted
from misc import load_model, evaluation_with_img_saving
from dataloader import denoiser_dataset_loader
from tqdm import tqdm
import AINDNet

#################################
### HARDCODED HYPERPARAMETERS ###
#################################

training_patch_size = 64
evaluation_batch_size  = 1

in_channels   = 3
out_channels  = 3

learning_rate = 1e-4

#################################
### HARDCODED HYPERPARAMETERS ###
#################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working with device:", device)

modelsavepath           = './trainedmodel.pth'
model = AINDNet.AINDNet(in_channels, out_channels)
model = model.to(device).float()

optimizer  = torch.optim.Adam(model.parameters(),lr=learning_rate)

model, _ = load_model(model, optimizer, modelsavepath)
model    = model.to(device).float()

#### evaluation of mri images

result_path          = './predictions/mri/'
os.makedirs(result_path, exist_ok=True)

in_file_path_valid         = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/mri_images_gt/bt_noisy_test/03/'
out_file_path_valid        = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/mri_images_gt/bt_gt_test/'
valid_loader = denoiser_dataset_loader(in_path = in_file_path_valid, out_path = out_file_path_valid, 
                                 bs=evaluation_batch_size, ps=training_patch_size, trn_flag=False, shuffle_flag=False)

img_names = natsorted(sorted(glob.glob(in_file_path_valid + "/*.jpg"), key=len))
evaluation_with_img_saving(model, valid_loader, result_path, img_names, device, crop=False)


######### only for testing the generalizability of the model
########## ct data evaluation
result_path_ct          = './predictions/ct/'
os.makedirs(result_path_ct, exist_ok=True)

in_file_path_valid_ct         = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/ct_noisy/03/'
out_file_path_valid_ct        = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/ct_images_gt/'
valid_loader_ct = denoiser_dataset_loader(in_path = in_file_path_valid_ct, out_path = out_file_path_valid_ct, 
                                 bs=evaluation_batch_size, ps=training_patch_size, trn_flag=False, shuffle_flag=False)
                                
img_names = natsorted(sorted(glob.glob(in_file_path_valid_ct + "/*.jpg"), key=len))
evaluation_with_img_saving(model, valid_loader_ct, result_path_ct, img_names, device, crop=True)


######### lymphocytes data evaluation
result_path_lym          = './predictions/lymphocytes/'
os.makedirs(result_path_lym, exist_ok=True)

in_file_path_valid_lym         = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/lymphocytes_noisy/03/'
out_file_path_valid_lym        = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/lymphocytes_images_gt/'
valid_loader_lym = denoiser_dataset_loader(in_path = in_file_path_valid_lym, out_path = out_file_path_valid_lym, 
                                 bs=evaluation_batch_size, ps=training_patch_size, trn_flag=False, shuffle_flag=False)
                                
img_names = natsorted(sorted(glob.glob(in_file_path_valid_lym + "/*.jpg"), key=len))
evaluation_with_img_saving(model, valid_loader_lym, result_path_lym, img_names, device, crop=False)
