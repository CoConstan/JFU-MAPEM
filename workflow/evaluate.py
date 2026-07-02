import torch
import numpy as np
from pathlib import Path
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.io.SPECT import dicom
from utils.data_utils import HDF5Dataset, save_torch_tensor
from torch.utils.data import DataLoader
from utils.config_utils import read_cli, read_config
from models.U_MAPEM import U_MAPEM, U_MAPEM_JFB
from models.U_BSREM import U_BSREM, U_BSREM_JFB


from utils.DL_utils import NRMSE, RMSE, PSNR, NMAE

__VERSION__ = """
RELEASE
=======
    Name    : evaluate.py
    Release : 0.0.4
    
"""

"""
------------------------------------------------------------------------------------------------------------------------
                                                     Workflow
------------------------------------------------------------------------------------------------------------------------
"""
print(__VERSION__)

cli = read_cli(__VERSION__)
config = read_config(cli.config_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = Path(config["io"]["data_path"])
save_path = Path(config["io"]["output_path"]) / config["io"]["experiment_name"]
save_path.mkdir(parents=True, exist_ok=True)

im_path = save_path/ "images"
im_path.mkdir(parents=True, exist_ok=True)

#Create Dataloader
print("Creating Dataloader")
dataset = HDF5Dataset(data_path/'test.h5')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers= 0)

#Make metadata
object_meta = SPECTObjectMeta(dr = config["image_metadata"]["spacing"], shape= config["image_metadata"]["size"])
proj_meta = SPECTProjMeta(projection_shape= config["proj_metadata"]["size"], dr= config["proj_metadata"]["spacing"], angles= np.linspace(0, 360, config["proj_metadata"]["n_angles"], endpoint=False).tolist(), radii=[config["proj_metadata"]["radii"]] * config["proj_metadata"]["n_angles"])

psf_meta = dicom.get_psfmeta_from_scanner_params('SY-ME', energy_keV=208, intrinsic_resolution=0.38)
psf_meta.sigma_fit_params = [np.float64(0.03235363042582603), np.float64(0.11684338873367237), np.float64(0)]

psf_transform = SPECTPSFTransform(psf_meta)

#Create model
print("Creating Model")
if config["model"]["name"] == "U_MAPEM":
    model = U_MAPEM(beta = config["model"]["beta"], n_iter = config["model"]["n_iter"])
    model.float().to(device)
    #print the number of parameters
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")

elif config["model"]["name"] == "U_MAPEM_JFB":
    model = U_MAPEM_JFB(beta = config["model"]["beta"], n_iter = config["model"]["n_iter"])
    model.float().to(device)
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")

elif config["model"]["name"] == "U_BSREM":
    model = U_BSREM(beta = config["model"]["beta"], n_iter = config["model"]["n_iter"])
    model.float().to(device)
    model.alpha_num = config["model"]["alpha_num"]
    model.alpha_denom = config["model"]["alpha_denom"]
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")

elif config["model"]["name"] == "U_BSREM_JFB":
    model = U_BSREM_JFB(beta = config["model"]["beta"], n_iter = config["model"]["n_iter"])
    model.alpha_num = config["model"]["alpha_num"]
    model.alpha_denom = config["model"]["alpha_denom"]
    model.float().to(device)
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")

else:
    raise ValueError("Model not implemented yet, available models are : U_MAPEM, U_MAPEM_JFB, U_BSREM, U_BSREM_JFB")

#Load model
print("Loading Model")
model.load_state_dict(torch.load(config["io"]["model_path"]))

#Evaluation
print("Evaluation")
model.eval()

nrmse_out = []
nrmse_RM = []

rmse_out, rmse_RM= [], []
psnr_out, psnr_RM= [], []
nmae_out, nmae_RM = [], []


with torch.no_grad():
    for batch in dataloader:
        print("------------------------------------------")
        print(batch["id"][0])
        source = batch["source"].squeeze().float().to(device)
        attmap = batch["attmap"].squeeze().float().to(device)
        proj = batch["proj"].squeeze().float().to(device)
        Rm = batch["RM"].squeeze().float().to(device)

        attenuation_transform = SPECTAttenuationTransform(attmap)
        spect_sys = SPECTSystemMatrix(
            obj2obj_transforms=[attenuation_transform, psf_transform],
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta
        )
        model.set_Asum(spect_sys, proj.shape)
        model.eval()
        out = model(Rm, proj, spect_sys)

        patient_path = im_path / batch["id"][0]
        patient_path.mkdir(parents=True, exist_ok=True)

        save_torch_tensor(source, patient_path / "source.nii", config["image_metadata"]["spacing"], transpose=True)
        save_torch_tensor(out, patient_path / "out.nii", config["image_metadata"]["spacing"], transpose=True)
        save_torch_tensor(Rm, patient_path / "OSEM.nii", config["image_metadata"]["spacing"], transpose=True)
        save_torch_tensor(attmap, patient_path / "attmap.nii", config["image_metadata"]["spacing"], transpose=True)

        # save segmentation masks

        if config["eval"]["save_masks"]:
            segmentation_path = patient_path / 'segmentation'
            segmentation_path.mkdir(parents=True, exist_ok=True)
            for mask_name in batch["masks_name"]:
                mask = batch["masks"][batch["masks_name"].index(mask_name)].squeeze(0).float()
                save_torch_tensor(mask, segmentation_path / f"{mask_name[0]}.nii.gz", config["image_metadata"]["spacing"], transpose=True)

        # Calculate global metrics
        if config['eval']["verbose"]:
            print(f"model/source | NRMSE : {NRMSE(out, source)} | RMSE : {RMSE(out, source)}, PSNR : {PSNR(out, source)}, NMAE : {NMAE(out, source)}")
            print(f"RM/source | NRMSE : {NRMSE(Rm, source)} | RMSE : {RMSE(Rm, source)}, PSNR : {PSNR(Rm, source)}, NMAE : {NMAE(Rm, source)}")
        
        nrmse_out.append(NRMSE(out, source))
        nrmse_RM.append(NRMSE(Rm, source))

        rmse_out.append(RMSE(out, source))
        rmse_RM.append(RMSE(Rm, source))

        psnr_out.append(PSNR(out, source))
        psnr_RM.append(PSNR(Rm, source))

        nmae_out.append(NMAE(out, source))
        nmae_RM.append(NMAE(Rm, source))



    #compute the average metrics and error

    nrmse_out_avg, nrmse_RM_avg = torch.mean(torch.tensor(nrmse_out)).cpu().numpy(), torch.mean(torch.tensor(nrmse_RM)).cpu().numpy()
    rmse_out_avg, rmse_RM_avg = torch.mean(torch.tensor(rmse_out)).cpu().numpy(), torch.mean(torch.tensor(rmse_RM)).cpu().numpy()
    psnr_out_avg, psnr_RM_avg = torch.mean(torch.tensor(psnr_out)).cpu().numpy(), torch.mean(torch.tensor(psnr_RM)).cpu().numpy()
    nmae_out_avg, nmae_RM_avg = torch.mean(torch.tensor(nmae_out)).cpu().numpy(), torch.mean(torch.tensor(nmae_RM)).cpu().numpy()
    
    print("--------------------Average Metrics with error----------------------")
    print(f"Average Metrics model/source | NRMSE: {nrmse_out_avg} | RMSE: {rmse_out_avg} | PSNR: {psnr_out_avg} | NMAE: {nmae_out_avg}")
    print(f"Average Metrics OSEM(10,8)/source | NRMSE: {nrmse_RM_avg} | RMSE: {rmse_RM_avg} | PSNR: {psnr_RM_avg} | NMAE: {nmae_RM_avg}")

print("Done")

