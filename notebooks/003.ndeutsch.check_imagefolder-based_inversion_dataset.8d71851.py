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

# # Checking the ImageFolder-based inversion data generation

# %load_ext autoreload
# %autoreload 2

import os
import torch
import PIL
from matplotlib import pyplot as plt
import src.models.dcgan32 as gan

# ## Oldest legacy: one file per image and label
# ### Dataset creation

import src.data.dcgan32.create_inversion_dataset as creation
os.chdir("/home/ndeutsch/face_gans/face_gans/")

creation.generate_dcgan32_inversion_dataset_many_files

# !rm -rf data/processed/tests/test1

creation.generate_dcgan32_inversion_dataset_many_files(dataset_root="data/processed/tests/test1",
                                                       dataset_size=510)

im1=PIL.Image.open("data/processed/tests/test1/imgs/1.png")
plt.figure(figsize=(2,2))
plt.imshow(im1)

# +
G=gan.DCGen32(load_weights=True)
G.eval()
z=torch.load("data/processed/tests/test1/vecs/1.pkl")
gim1 = G(z.unsqueeze(0)).squeeze().detach().permute(1,2,0)*0.5+0.5

plt.figure(figsize=(2,2))
plt.imshow(gim1)
# -

# ### Dataset loading

import src.data.dcgan32.load_inversion_dataset as loading

loading.create_dcgan32_inversion_dataloader_many_files

dataloader = loading.create_dcgan32_inversion_dataloader_many_files(root="data/processed/tests/test1")

sample = next(iter(dataloader))

im2 = sample[0][0]
z2 = sample[1][0]

plt.figure(figsize=(2,2))
plt.imshow(im2.permute(1,2,0)*0.5+0.5)

# +
G=gan.DCGen32(load_weights=True)
G.eval()
gim2 = G(z2.unsqueeze(0)).squeeze().detach().permute(1,2,0)*0.5+0.5

plt.figure(figsize=(2,2))
plt.imshow(gim2)
# -

# ## Slight improvement: all labels in memory
# ### Dataset creation

creation.generate_dcgan32_inversion_dataset_many_images_one_labelfile

# !rm -rf "data/processed/tests/test2"

creation.generate_dcgan32_inversion_dataset_many_images_one_labelfile(dataset_root="data/processed/tests/test2",
                                                                      dataset_size=510)

im1=PIL.Image.open("data/processed/tests/test2/imgs/1.png")
plt.figure(figsize=(2,2))
plt.imshow(im1)

# +
G=gan.DCGen32(load_weights=True)
G.eval()
zs=torch.load("data/processed/tests/test2/vecs/labels.pkl")
z=zs[1]
gim1 = G(z.unsqueeze(0)).squeeze().detach().permute(1,2,0)*0.5+0.5

plt.figure(figsize=(2,2))
plt.imshow(gim1)
# -

# ### Dataset loading

loading.create_dcgan32_inversion_dataloader_many_images

dataloader2=loading.create_dcgan32_inversion_dataloader_many_images(root="data/processed/tests/test2")

# +
sample=dataloader2.dataset[12]
im = sample[0]
z = sample[1]

plt.figure(figsize=(2,2))
plt.imshow(im.permute(1,2,0)*0.5+0.5)
plt.show()
G=gan.DCGen32(load_weights=True)
G.eval()
gim = G(z.unsqueeze(0)).squeeze().detach().permute(1,2,0)*0.5+0.5

plt.figure(figsize=(2,2))
plt.imshow(gim)
plt.show()

# +
batch=next(iter(dataloader2))
im = batch[0][0]
z = batch[1][0]

plt.figure(figsize=(2,2))
plt.imshow(im.permute(1,2,0)*0.5+0.5)
plt.show()
G=gan.DCGen32(load_weights=True)
G.eval()
gim = G(z.unsqueeze(0)).squeeze().detach().permute(1,2,0)*0.5+0.5

plt.figure(figsize=(2,2))
plt.imshow(gim)
plt.show()
# -


