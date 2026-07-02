import torch
import numpy as np
from pathlib import Path
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.transforms.SPECT import SPECTPSFTransform
from pytomography.io.SPECT import dicom
from torch.utils.data import DataLoader
from utils.config_utils import read_cli, read_config
from utils.DL_utils import train
from utils.data_utils import HDF5Dataset
from models.U_MAPEM import U_MAPEM, U_MAPEM_JFB
from models.U_BSREM import U_BSREM, U_BSREM_JFB


__VERSION__ = """
RELEASE
=======
    Name    : training.py
    Release : 1.0.3
    
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

#Create Dataloader
print("Creating Dataloader")
train_dataset = HDF5Dataset(data_path/'train.h5')
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers= 0)
val_dataset = HDF5Dataset(data_path/'val.h5')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers= 0)

#Make metadata
object_meta = SPECTObjectMeta(dr = config["image_metadata"]["spacing"], shape= config["image_metadata"]["size"])
proj_meta = SPECTProjMeta(projection_shape= config["proj_metadata"]["size"], dr= config["proj_metadata"]["spacing"], angles= np.linspace(0, 360, config["proj_metadata"]["n_angles"], endpoint=False).tolist(), radii=[config["proj_metadata"]["radii"]] * config["proj_metadata"]["n_angles"])

psf_meta = dicom.get_psfmeta_from_scanner_params('SY-ME', energy_keV=208, intrinsic_resolution=0.38)
psf_meta.sigma_fit_params = [np.float64(0.03235363042582603), np.float64(0.11684338873367237), np.float64(0)]
psf_transform = SPECTPSFTransform(psf_meta)

pytomo_args = (object_meta, proj_meta, psf_transform)

#Create model
print("Creating Model")
if config["model"]["name"] == "U_MAPEM":
    model = U_MAPEM(beta = config["model"]["beta"], n_iter = config["model"]["n_iter"])
    model.float().to(device)
    if config["training"]["start_from_checkpoint"]:
        model.load_state_dict(torch.load(config["training"]["checkpoint_path"]))
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")


elif config["model"]["name"] == "U_MAPEM_JFB":
    model = U_MAPEM_JFB(beta = config["model"]["beta"], n_iter = config["model"]["n_iter"])
    model.float().to(device)
    if config["training"]["start_from_checkpoint"]:
        model.load_state_dict(torch.load(config["training"]["checkpoint_path"]))
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")

elif config["model"]["name"] == "U_BSREM":
    model = U_BSREM(beta = config["model"]["beta"], n_iter = config["model"]["n_iter"])
    model.alpha_num = config["model"]["alpha_num"]
    model.alpha_denom = config["model"]["alpha_denom"]
    model.float().to(device)
    if config["training"]["start_from_checkpoint"]:
        model.load_state_dict(torch.load(config["training"]["checkpoint_path"]))
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")

elif config["model"]["name"] == "U_BSREM_JFB":
    model = U_BSREM_JFB(beta = config["model"]["beta"], n_iter = config["model"]["n_iter"])
    model.alpha_num = config["model"]["alpha_num"]
    model.alpha_denom = config["model"]["alpha_denom"]
    model.float().to(device)
    if config["training"]["start_from_checkpoint"]:
        model.load_state_dict(torch.load(config["training"]["checkpoint_path"]))
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")

else:
    raise ValueError("Model not implemented yet, available models are : U_MAPEM, U_MAPEM_JBF, U_BSREM, U_BSREM_JBF")

if config["training"]["optimizer"] == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
elif config["training"]["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
elif config["training"]["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=config["training"]["learning_rate"])
elif config["training"]["optimizer"] == "RMSprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["training"]["learning_rate"])

if config["training"]["loss_function"] == "MSELoss":
    criterion = torch.nn.MSELoss()
    model.grad_penalty = False
elif config["training"]["loss_function"] == "L1Loss":
    criterion = torch.nn.L1Loss()
    model.grad_penalty = False
else:
    print(f"{config["training"]["loss_function"]} not implemented.")

print("Experiment Name : ", config["io"]["experiment_name"])
print("Training Configuration : ")
print("Using Model : ", config["model"]["name"])
print("Using Loss Function : ", config["training"]["loss_function"])
print("Using Optimizer : ", config["training"]["optimizer"])
print("Learning Rate : ", config["training"]["learning_rate"])

#Training
print("Training")
results = train(train_dataloader, val_dataloader, model, optimizer, criterion, pytomo_args, config["training"], scheduler=None, verbose=1)

#Save model
print("Saving Model")
print("Model saved at : ", config["io"]["model_path"])
torch.save(results['best_model_state'], config["io"]["model_path"])
print("Training finished")

model_list = results['model_list']
for idx, mdl in enumerate(model_list):
    model_path = Path(config["io"]["model_path"]).parent / "in_training"/ config["model"]["name"] / config["io"]["experiment_name"]
    model_path.mkdir(parents=True, exist_ok=True)
    # save with epoch number 001, 002, ...
    torch.save(mdl, model_path / f"model_epoch_{idx+1:03d}.pt")
    print(f"Model at epoch {idx+1} saved at : ", model_path)

#Save results
results_path = Path(config["io"]["output_path"]) / config["io"]["experiment_name"]
results_path.mkdir(parents=True, exist_ok=True)

# Save updated arrays
np.save( results_path / "train_losses.npy", results['train_losses'])
np.save( results_path / "val_losses.npy", results['val_losses'])

print("Results saved at : ", results_path)

#Save config
config_path = results_path / "config.json"
with open(config_path, 'w') as f:
    import json
    json.dump(config, f, indent=4)



