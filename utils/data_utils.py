import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

def save_torch_tensor(tensor, save_path, spacing, origin = None, transpose=True):
    """
    Save a torch tensor as an MHD file with the specified spacing.
    """
    if transpose:
        tensor = tensor.permute(2, 1, 0)
    tensor_np = tensor.cpu().detach().numpy()
    itk_image = sitk.GetImageFromArray(tensor_np)
    itk_image.SetSpacing(spacing)
    if origin is not None:
        itk_image.SetOrigin(origin)
    sitk.WriteImage(itk_image, save_path)

def open_mhd(image_path):
    itk_image = sitk.ReadImage(image_path)
    np_image = sitk.GetArrayFromImage(itk_image)
    torch_image = torch.from_numpy(np_image).to(dtype=torch.float32)
    
    size = np_image.shape
    spacing = np.array(itk_image.GetSpacing())
    origin = np.array(itk_image.GetOrigin())

    return torch_image, size, spacing, origin

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, n_patient=None):
        """
        Custom PyTorch dataset for reading HDF5 data.
        Args:
            h5_path (str): Path to the HDF5 file.
        """
        self.h5_path = h5_path
        
        # Get a list of patient groups
        with h5py.File(self.h5_path, "r") as h5_file:
            self.patients = list(h5_file.keys())[:n_patient]  # ['patient0', 'patient1', ...]

    def __len__(self):
        """Returns the number of patients."""
        return len(self.patients)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as file:  # Open file per call (worker-safe)
            patient_id = self.patients[idx]
            patient_group = file[patient_id]

            source = torch.tensor(patient_group["source"][:], dtype=torch.float32)
            attmap = torch.tensor(patient_group["attmap"][:], dtype=torch.float32)
            proj = torch.tensor(patient_group["proj"][:], dtype=torch.float32)
            Rm = torch.tensor(patient_group["RM"][:], dtype=torch.float32)

            name_liste = []
            masks_list = []
            if "segmentation" in patient_group:
                name_liste = list(patient_group["segmentation"].keys())
                masks_list = []
                for mask_name in name_liste:
                    mask = torch.tensor(patient_group["segmentation"][mask_name][:], dtype=torch.float32)
                    masks_list.append(mask)

        return {"source": source, "attmap": attmap, "proj": proj, "id": patient_id, "RM": Rm, "masks_name":name_liste, "masks": masks_list}

