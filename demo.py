#%%
from pathlib import Path
import torch
from pytomography.projectors.SPECT import SPECTSystemMatrix
from utils.demo_utils import open_demo_data, load_model, osem_reconstruction, mapem_reconstruction
from utils.plot_utils import plot
import SimpleITK as sitk
# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load model
print("Load Model")
model_name = "U_MAPEM_JFB"
model = load_model(model_name, n_iter=18, device=device)

#%%
print("Demo data")
data_path = Path('Data/IEC_EXP')

proj, scatter, object_meta, proj_meta, attenuation_transform, psf_transform = open_demo_data(data_path)


#%%
source_1mm = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(data_path/"source_1mm.mha")))
plot(source_1mm, 180, "Source (1mm)", units='kBq/mL', cmap='hot', axes=0, transpose=False)

source_4mm = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(data_path/"source_4mm.mha"))).to(device)
plot(source_4mm, 69, "Source (1mm)", units='kBq/mL', cmap='hot', axes=0, transpose=False)

attmap = attenuation_transform.attenuation_map
plot(attmap, 69, "Attmap (4mm)", cmap='grey', axes=0)


#%%
spect_sys = SPECTSystemMatrix(
    obj2obj_transforms=[attenuation_transform,psf_transform],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta
    )


#%%
plot(proj,30,"Projections", axes=2)


#%%
print("OSEM reconstruction")

osem_rec = osem_reconstruction(proj, spect_sys, n_iter=10,subset=8) 
plot(osem_rec, 69, "OSEM reconstruction", units="kBq/mL", axes=0)

#%%
print("MAPEM-OSL reconstruction")
mapem_rec = mapem_reconstruction(proj, attmap, spect_sys, n_iter=20,subset=8)
plot(mapem_rec, 69, "MAPEM-OSL reconstruction", units="kBq/mL", axes=0)


#%%
#Evaluation
print("Model reconstruction")
model.eval()

with torch.no_grad():
    model.set_Asum(spect_sys, proj.shape)
    model.eval()
    rec = model(osem_rec, proj, spect_sys)

plot(rec, 69, "Model reconstruction", units="kBq/mL", axes=0)

# %%
