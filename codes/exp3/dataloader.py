import glob, torch, imageio
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def read_image(provided_list, index):
    input_img = imageio.imread(provided_list[index])     
    input_img = torch.from_numpy((input_img) / 255.0).permute(2,0,1).float()
    return input_img


class DenoiserDataset(Dataset):
    def __init__(self, input_file_path, output_file_path, patch_size, train_flag):

        self.input_imgs_path     = input_file_path
        self.input_imgs          = natsorted(sorted(glob.glob(self.input_imgs_path + "/*.jpg"), key=len))

        self.output_imgs_path    = output_file_path
        self.output_imgs         = natsorted(sorted(glob.glob(self.output_imgs_path + "/*.jpg"), key=len))

        self.ps    = patch_size
        self.train = train_flag
		    

    def __len__(self):
        return len(self.input_imgs)

    def __getitem__(self, index):

        input_img            = imageio.imread(self.input_imgs[index])
        input_img            = torch.from_numpy((input_img) / 255.0).permute(2,0,1).float()  
        c, h, w = input_img.shape	

        output_img           = imageio.imread(self.output_imgs[index])
        output_img           = torch.from_numpy((output_img) / 255.0).permute(2,0,1).float()

        if self.train:
            random_row_index = np.random.randint(0,h-self.ps)
            random_col_index = np.random.randint(0,w-self.ps)

            input_img             = input_img[:, random_row_index:random_row_index+self.ps, random_col_index:random_col_index+self.ps]
            output_img            = output_img[:, random_row_index:random_row_index+self.ps, random_col_index:random_col_index+self.ps]

        return input_img, output_img

def denoiser_dataset_loader(in_path, out_path, bs, ps, trn_flag, shuffle_flag, drop_last_batch=False):
    ds     = DenoiserDataset(input_file_path=in_path, output_file_path=out_path, patch_size=ps, train_flag=trn_flag)
    loader = DataLoader(dataset = ds, batch_size = bs, shuffle=shuffle_flag, drop_last=drop_last_batch)
    return loader
