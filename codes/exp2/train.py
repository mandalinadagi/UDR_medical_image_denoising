import numpy as np 
import torch
import torch.nn as nn
import os, logging, piq
from natsort import natsorted
from misc import load_model, save_model, parameters_count, evaluation, evaluation_with_img_saving, seed_torch
from dataloader import denoiser_dataset_loader
from tqdm import tqdm
import DnCNN

#################################
### HARDCODED HYPERPARAMETERS ###
#################################

training_patch_size = 64
training_batch_size    = 8
evaluation_batch_size  = 1

channels   = 3

learning_rate       = 4e-4
total_epochs        = 101
eval_every_x_epochs = 25

lr_halving_1_at_epoch = 75
lr_halving_2_at_epoch = 85

#################################
### HARDCODED HYPERPARAMETERS ###
#################################


modelsavepath           = './trainedmodel.pth'

in_file_path_train         = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/mri_images_gt/bt_noisy_train/03/'
out_file_path_train        = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/mri_images_gt/bt_gt_train/'

in_file_path_valid         = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/mri_images_gt/bt_noisy_test/03/'
out_file_path_valid        = '/kuacc/users/ckorkmaz14/UDR_medical_image_denoising/data/mri_images_gt/bt_gt_test/'

logging.basicConfig(filename= "training.log",level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working with device:", device)
seed_torch()

result_path          = './predictions/mri/'
os.makedirs(result_path, exist_ok=True)

train_loader = denoiser_dataset_loader(in_path = in_file_path_train, out_path = out_file_path_train, 
                                 bs=training_batch_size, ps=training_patch_size, trn_flag=True, shuffle_flag=True)

valid_loader = denoiser_dataset_loader(in_path = in_file_path_valid, out_path = out_file_path_valid, 
                                 bs=evaluation_batch_size, ps=training_patch_size, trn_flag=False, shuffle_flag=False)


model = DnCNN.DnCNN(channels)
model = model.to(device).float()

total_parameters, trainable_params = parameters_count(model)
logging.info("Total Parameters: "+ str(total_parameters))
logging.info("Number of Trainable Parameters: "+ str(trainable_params))

l1_loss    = nn.L1Loss()
optimizer  = torch.optim.Adam(model.parameters(),lr=learning_rate)

starting_test_loss = evaluation(model, valid_loader, l1_loss, device)
logging.info("Starting Test Loss: "+ str(starting_test_loss))
print(starting_test_loss)

for epoch in range(total_epochs):
    model.train()
    print("")
    print("Epoch:", epoch)
    print("")

    if epoch == lr_halving_1_at_epoch:
        learning_rate /= 2
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
    if epoch == lr_halving_2_at_epoch:
        learning_rate /= 2
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
 
    for i, (input_img, output_img) in enumerate(tqdm(train_loader)):

        input_img             = input_img.to(device).float()
        output_img            = output_img.to(device).float()

        img_prediction = model(input_img)

        loss_l1       = l1_loss(img_prediction, output_img)
        loss          = loss_l1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info("Epoch:   "+ str(epoch))  
    logging.info("Loss L1 Img: "+ str(loss_l1)) 
    logging.info("Last LR: "+ str(learning_rate))

    if((epoch % eval_every_x_epochs) == 0):
        test_loss = evaluation(model, valid_loader, l1_loss, device)
        logging.info("Eval Epoch: " + str(epoch))
        logging.info("Test Loss:  " + str(test_loss))

        if test_loss < starting_test_loss:
            save_model(model, optimizer, modelsavepath)
            starting_test_loss = test_loss
            logging.info("-----------------New Best----------------")

