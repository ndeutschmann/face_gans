# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # HDF5 datasets

import os
import torch
import tables
import PIL
from matplotlib import pyplot as plt
import src.models.dcgan32 as gan
import src.data.dcgan32.create_inversion_dataset as ds_create
import src.models.dcgan32 as gan

import src.data.datasets.indexable_dataset

os.chdir("/home/ndeutsch/face_gans/face_gans/")

# ## Dataset creation

ds_create.generate_dcgan32_inversion_dataset_two_h5_tables

# !rm -rf data/processed/tests/test3/

ds_create.generate_dcgan32_inversion_dataset_two_h5_tables('data/processed/tests/test3/',
                                                    dataset_size=510)

# +
imagefile = tables.open_file('data/processed/tests/test3/images.h5',mode="r")
images = imagefile.get_node("/data/images")

labelfile = tables.open_file('data/processed/tests/test3/labels.h5',mode="r")
labels = labelfile.get_node("/data/labels")
# -

im = torch.tensor(images[12])
z = torch.tensor(labels[12])

# +
plt.figure(figsize=(2,2))
plt.imshow(im.permute(1,2,0)*0.5+0.5)
plt.show()

G=gan.DCGen32(load_weights=True)
G.eval()
gim1 = G(z.unsqueeze(0)).squeeze().detach().permute(1,2,0)*0.5+0.5

plt.figure(figsize=(2,2))
plt.imshow(gim1)
# -

# ### Checking the general pytables importer

import src.data.datasets.indexable_dataset as dsload
import torch
from importlib import reload
#reload(dsload)

# +
files= ["data/processed/tests/test3/images.h5","data/processed/tests/test3/labels.h5"]
arrays = ["/data/images","/data/labels"]

ds = dsload.MultiHDF5TablesDataset(files,arrays)

# +
im,z = ds[14]
im = torch.tensor(im) 
z = torch.tensor(z)

plt.figure(figsize=(2,2))
plt.imshow(im.permute(1,2,0)*0.5+0.5)
plt.show()

G=gan.DCGen32(load_weights=True)
G.eval()
gim1 = G(z.unsqueeze(0)).squeeze().detach().permute(1,2,0)*0.5+0.5

plt.figure(figsize=(2,2))
plt.imshow(gim1)
# -

import src.data.dcgan32.load_inversion_dataset as loader
#reload(loader)

dataloader = loader.create_dcgan32_inversion_dataloader_hdf5_tables(root="data/processed/tests/test3")

bimage,bz=next(iter(dataloader))

# +
im = bimage[17]
z = bz[17]

plt.figure(figsize=(2,2))
plt.imshow(im.permute(1,2,0)*0.5+0.5)
plt.show()

G=gan.DCGen32(load_weights=True)
G.eval()
gim1 = G(z.unsqueeze(0)).squeeze().detach().permute(1,2,0)*0.5+0.5

plt.figure(figsize=(2,2))
plt.imshow(gim1)
# -


