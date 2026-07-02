import torch
import matplotlib.pyplot as plt

def plot(img, slice, title, units = 'Counts', axes=1, cmap='hot', transpose = True):
    if transpose:
        img = torch.permute(img, (2, 1, 0))
        # img = torch.flip(img, [0])

    plt.figure(figsize=(6, 6))

    if axes == 0:
        im = plt.imshow(img[slice, :, :].cpu(), cmap=cmap)
    elif axes == 1:
        im = plt.imshow(img[:, slice, :].cpu(), cmap=cmap)
    elif axes == 2:
        im = plt.imshow(img[:, :, slice].cpu(), cmap=cmap)
    else:
        raise ValueError("axes must be 0, 1, or 2")

    cbar = plt.colorbar(im)
    cbar.set_label(units) 
    plt.title(title)
    plt.show()