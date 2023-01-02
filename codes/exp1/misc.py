import random, numpy, torch, os, imageio, logging   
from tqdm import tqdm
import matplotlib.pyplot as plt

def seed_torch(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, optimizer, saving_path):
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state,saving_path)

def load_model(model, optimizer, saving_path):
    checkpoint = torch.load(saving_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def parameters_count(model):
    total = sum([i.numel() for i in model.parameters()])
    trainable = sum([i.numel() for i in model.parameters() if i.requires_grad])
    print("Total parameters : {}. Trainable parameters : {}".format(total, trainable))
    return total, trainable

def evaluation(model, valid_loader, l1_loss, tv_loss, device):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (input_img, output_img) in enumerate(tqdm(valid_loader)):

            input_img       = input_img.to(device).float()                   
            output_img      = output_img.to(device).float()
            mask_prediction, img_prediction = model(input_img)

            loss_tv_mask  = tv_loss(mask_prediction)
            loss_l1       = l1_loss(img_prediction, output_img)
            loss          = loss_tv_mask + loss_l1
            test_loss       += loss


        test_loss /= len(valid_loader)
    return test_loss

def evaluation_with_img_saving(model, valid_loader, result_path, img_names, device, crop):
    model.eval()
    with torch.no_grad():
        for i, (input_img, output_img) in enumerate(tqdm(valid_loader)):
            if crop: 
                input_img = input_img[:,:,:512,:512]
                output_img = output_img[:,:,:512,:512]

            input_img       = input_img.to(device).float()                   
            output_img      = output_img.to(device).float()

            mask_prediction, img_prediction = model(input_img)

            img_prediction  = img_prediction[0].permute(1,2,0).detach().cpu().numpy()
            img_prediction_out = numpy.round((img_prediction*255)).clip(0,255).astype('uint8')

            img_name   = (img_names[i].split('/')[-1]).split(".")[0] 
            imageio.imsave(result_path + img_name +'_UDR.png', img_prediction_out)


# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.imsave("a.png")
