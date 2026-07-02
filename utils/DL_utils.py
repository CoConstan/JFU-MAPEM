import torch
import time
import copy
from tqdm import tqdm
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.transforms.SPECT import SPECTAttenuationTransform
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.algorithms.preconditioned_gradient_ascent import OSEM
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.autograd as autograd

def RMSE(img, ref):
    rmse = torch.sqrt(torch.mean((ref - img) ** 2))
    return rmse

def NRMSE(img, ref):
    rmse = torch.sqrt(torch.mean((ref - img) ** 2))
    return rmse / torch.mean(torch.abs(ref))

def NMAE(img, ref):
    mae = torch.mean(torch.abs(ref - img))
    return mae / torch.mean(torch.abs(ref))

def PSNR(img, ref):
    L = ref.max() - ref.min()
    mse = torch.mean((ref - img) ** 2)
    PSNR = 10 * torch.log10(L**2 / mse)
    return PSNR

def CNR(mask1, mask2, img):
    mu1 = torch.mean(img[mask1])
    mu2 = torch.mean(img[mask2])
    std1 = torch.std(img[mask1])
    std2 = torch.std(img[mask2])
    CNR = (mu1 - mu2) / std2
    return CNR

def VAA(mask, img, src, percent=5):
    mask = mask.bool()
    img_masked = img[mask].float()
    src_masked = src[mask].float()

    relative_error = torch.abs(src_masked - img_masked) / (src_masked + 1e-8)
    return (relative_error < percent / 100).float().mean()


def RC(mask: torch.Tensor, img: torch.Tensor, src: torch.Tensor) -> float:
    """
    Calculate Recovery Coefficient (RC) using masked slicing.

    Args:
        img (torch.Tensor): Measured image.
        src (torch.Tensor): Ground truth/source image.
        mask (torch.Tensor): Boolean mask (same shape as img/src) indicating ROI.

    Returns:
        float: Recovery coefficient value.
    """
    # Ensure inputs are float and mask is boolean
    img = img.float()
    src = src.float()
    mask = mask.bool()

    # Slice using the mask
    measured_vals = img[mask]
    true_vals = src[mask]

    true_sum = torch.sum(true_vals)

    rc = torch.sum(measured_vals) / (true_sum + 1e-8)
    return rc


def local_RMSE(mask, img, src):
    return torch.sqrt(torch.sum((img[mask] - src[mask])**2) / torch.sum(mask))      # A verifié 

def local_NMAE(mask, img, src):
    return torch.mean(torch.abs(img[mask] - src[mask])) / torch.mean(torch.abs(src[mask]))

def RMS(mask, img):
    return torch.std(img[mask]) / torch.mean(img[mask])


def evaluate(dataloader, model, criterion, pytomo_args, args):
    
    object_meta = pytomo_args[0]
    proj_meta = pytomo_args[1]
    psf_transform = pytomo_args[2]
    loss = 0
    outputs = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            source = batch["source"].squeeze().float().to(args["device"])
            attmap = batch["attmap"].squeeze().float().to(args["device"])
            proj = batch["proj"].squeeze().float().to(args["device"])
            Rm = batch["RM"].squeeze().float().to(args["device"])

            attenuation_transform = SPECTAttenuationTransform(attmap)
            spect_sys = SPECTSystemMatrix(
                obj2obj_transforms=[attenuation_transform, psf_transform],
                proj2proj_transforms=[],
                object_meta=object_meta,
                proj_meta=proj_meta
            )
            model.set_Asum(spect_sys, proj.shape)
            if model.ct_use is not None:
                model.ct = attmap
            out = model(Rm, proj, spect_sys)
            if model.grad_penalty == False:
                loss_ = criterion(out, source)
            else:
                loss_ = criterion(model.net, out, source)
            loss += loss_.item()
            outputs.append(out)
    
    model.train()
    val_loss = loss / len(dataloader)
    return val_loss, outputs

def train(train_loader, val_loader, model, optimizer, criterion, pytomo_args, args, scheduler = None, verbose = 0):
    """
    This function trains the model on the train_loader and evaluates it on the val_loader at each epoch.

    Args:
        train_loader (pytorch Dataloader): Dataloader that will be used to feed the model during training
        val_loader (pytorch Dataloader): Dataloader that will be used to feed the model during evaluation
        model (nn.Module): model to train
        optimizer (torch.optim): optimizer that will be used to update the model's weights (Adam, SGD...)
        criterion (torch.nn): criterion that will be used to compute the loss (CrossEntropy, MSELoss...)
        args (dict): arguments of the training (device, epochs, batch size, patience...)
        scheduler (_type_): scheduler that will be used to update the learning rate, None by default
        verbose (int, optional): set to 1 to print the loss and accuracy at each epoch. Defaults to 1. 

    Returns:
            'best_model_state'
            'train_losses'
            'val_losses'
            'duration'
    """

    object_meta = pytomo_args[0]
    proj_meta = pytomo_args[1]
    psf_transform = pytomo_args[2]
    model_list = []

    train_losses = []
    val_losses = []

    time_start = time.time()

    best_val_loss = float("inf")
    if args["start_from_checkpoint"]:
        best_val_loss, _, _ = evaluate(val_loader, model, criterion, pytomo_args, args)[0]
        best_model = copy.deepcopy(model.state_dict())
        
    consecutive_no_improvement = 0
    tot_epoch = args["epochs"]
    patience = args["patience"]

    for epoch in range(args["epochs"]):
        loop = tqdm(train_loader)
        train_loss = 0
        model.train()
        for i, batch in enumerate(loop):
            source = batch["source"].squeeze().float().to(args["device"])
            attmap = batch["attmap"].squeeze().float().to(args["device"])
            proj = batch["proj"].squeeze().float().to(args["device"])
            Rm = batch["RM"].squeeze().float().to(args["device"])

            attenuation_transform = SPECTAttenuationTransform(attmap)
            spect_sys = SPECTSystemMatrix(
                obj2obj_transforms=[attenuation_transform, psf_transform],
                proj2proj_transforms=[],
                object_meta=object_meta,
                proj_meta=proj_meta
            )
            model.set_Asum(spect_sys, proj.shape)       
            
            optimizer.zero_grad()
            if model.ct_use is not None:
                model.ct = attmap
            out = model(Rm, proj, spect_sys)
            loss = criterion(out, source)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            train_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            loop.set_description(f"Epoch {epoch+1}/{tot_epoch}")
            loop.set_postfix(_Loss_=(train_loss/(i+1)))


            
        train_loss = train_loss/len(train_loader)
        if verbose == 1:
            print("\t Train loss : ", train_loss)    
        train_losses.append(train_loss)
        
        model.eval()
        # Evaluate the model on the validation set 
        val_loss, _ = evaluate(val_loader, model, criterion, pytomo_args, args)
        if verbose == 1:
            print("\t Validation loss : ", val_loss)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            if verbose == 1:
                print(f"Validation loss improved from {best_val_loss} to {val_loss}")
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= patience:
            print(f"Early stopping after {consecutive_no_improvement} epochs of no improvement.")
            tot_epoch = epoch
            break

        # handle NAN values in losses
        if train_loss != train_loss or val_loss != val_loss:
            print("Loss is NaN, stopping training.")
            tot_epoch = epoch
            break
        
        model_list.append(copy.deepcopy(model.state_dict()))

    
    duration = time.time() - time_start
    print('Finished Training in:', duration, 'seconds with mean epoch duration:', duration/tot_epoch, ' seconds')
    results = {
        'best_model_state': best_model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'duration': duration,
        'model_list': model_list,
    }

    return results
