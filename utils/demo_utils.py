import numpy as np
import torch
import os
import SimpleITK as sitk
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.io.SPECT import dicom
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from models.U_MAPEM import U_MAPEM, U_MAPEM_JFB
from models.U_BSREM import U_BSREM, U_BSREM_JFB
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms.preconditioned_gradient_ascent import OSEM, OSMAPOSL
from pytomography.priors import RelativeDifferencePrior, TopNAnatomyNeighbourWeight


def open_demo_data(data_path): 
    path_CT = data_path / 'CT'
    files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
    file_NM = data_path / 'projection_data.dcm'
    photopeak = dicom.get_projections(file_NM, index_peak=0)

    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
    scatter = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)
    
    att_transform = SPECTAttenuationTransform(filepath=files_CT)
    att_transform.configure(object_meta, proj_meta)

    psf_meta =  dicom.get_psfmeta_from_scanner_params('SY-ME', energy_keV=208, intrinsic_resolution=0.38)
    # psf_meta.sigma_fit_params = [np.float64(0.03235363042582603), np.float64(0.11684338873367237), np.float64(0)]
    psf_transform = SPECTPSFTransform(psf_meta)

    return photopeak, scatter, object_meta, proj_meta, att_transform, psf_transform

def generate_metadata(spacing_img, shape_img, spacing_proj, shape_proj, n_angles, radii):
    object_meta = SPECTObjectMeta(dr = spacing_img, shape= shape_img)
    proj_meta = SPECTProjMeta(projection_shape= shape_proj, dr= spacing_proj, angles= np.linspace(0, 360, n_angles, endpoint=False).tolist(), radii= [radii] * n_angles)

    psf_meta = dicom.get_psfmeta_from_scanner_params('SY-ME', energy_keV=208, intrinsic_resolution=0.38)
    psf_meta.sigma_fit_params = [np.float64(0.03235363042582603), np.float64(0.11684338873367237), np.float64(0)]

    return object_meta, proj_meta, psf_meta


def load_model(model_name, n_iter, device, beta = 0.5):
    if model_name == "U_MAPEM":
        model = U_MAPEM(beta = beta, n_iter = n_iter)
        model.float().to(device)
        #print the number of parameters
        print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")
        model.load_state_dict(torch.load("weights/MAPEM/U_MAPEM.pth"))

    elif model_name == "U_MAPEM_JFB":
        model = U_MAPEM_JFB(beta = beta, n_iter = n_iter)
        model.float().to(device)
        print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")
        model.load_state_dict(torch.load(f"weights/MAPEM/U_MAPEM_JFB_{n_iter}it.pth"))

    elif model_name == "U_BSREM":
        model = U_BSREM(beta = beta, n_iter = n_iter)
        model.float().to(device)
        print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")
        model.load_state_dict(torch.load("weights/BSREM/U_BSREM.pth"))

    elif model_name == "U_BSREM_JFB":
        model = U_BSREM_JFB(beta = beta, n_iter = n_iter)
        model.float().to(device)
        print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")
        model.load_state_dict(torch.load(f"weights/BSREM/U_BSREM_JFB_{n_iter}it.pth"))

    else:
        raise ValueError("Model not implemented yet, available models are : U_MAPEM, U_MAPEM_JBF, U_BSREM, U_BSREM_JBF")

    return model

def osem_reconstruction(proj, spect_sys, n_iter, subset):
    lklhd = PoissonLogLikelihood(spect_sys, proj)
    osem = OSEM(likelihood=lklhd)
    rec = osem(n_iter, subset)
    return rec

def mapem_reconstruction(proj, attmap, spect_sys, n_iter, subset, gamma = 4.98, beta = 0.99):
    lklhd = PoissonLogLikelihood(spect_sys, proj)

    weight_top8anatomy = TopNAnatomyNeighbourWeight(attmap, N_neighbours=8)
    prior = RelativeDifferencePrior(beta=beta, gamma=gamma, weight=weight_top8anatomy)
    mapem = OSMAPOSL(likelihood=lklhd, prior=prior)
    rec = mapem(n_iter, subset)
    return rec