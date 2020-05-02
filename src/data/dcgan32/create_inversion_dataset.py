import os
import numpy as np
import tables
import torch
import torchvision
from src.models.dcgan32 import DCGen32

class ImagePerFileBatchSaver:
    def __init__(self,image_dir,ext="png",transform=None):
        self.image_dir = image_dir
        self.ext = ext
        self.transform = transform
        os.makedirs(image_dir, exist_ok=False)

    def save_batch(self,batch,first_index=0):
        for i, b in enumerate(batch):
            b_im = self.transform(b) if self.transform is not None else b
            b_im.save(os.path.join(self.image_dir, "{}.{}".format(i + first_index,self.ext)))

    def finalize(self):
        pass

class LabelPerTorchFileBatchSaver:
    def __init__(self,label_dir,ext="pkl",transform=None):
        self.label_dir = label_dir
        self.ext = ext
        self.transform = transform
        os.makedirs(label_dir, exist_ok=False)

    def save_batch(self, batch, first_index=0):
        for i,x in enumerate(batch):
            x_t = self.transform(x) if self.transform is not None else x
            torch.save(x_t, os.path.join(self.label_dir, "{}.{}".format(i + first_index,self.ext)))

    def finalize(self):
        pass


class LabelOneTorchFileBatchSaver:
    def __init__(self, label_dir, database_size, label_shape, filename="labels.pkl", transform=None):
        self.label_dir = label_dir
        self.filename = filename
        self.transform = transform
        self.labels = torch.zeros(database_size,*label_shape)
        os.makedirs(label_dir, exist_ok=False)

    def save_batch(self,batch,first_index=0):
        self.labels[first_index:first_index+batch.shape[0]] = batch

    def finalize(self):
        torch.save(self.labels, os.path.join(self.label_dir,self.filename))


class HDF5TensorSaver:
    def __init__(self,data_dir,*,
                 filename="images.h5",
                 database_size,
                 data_shape=(32, 32, 3),
                 group="data",
                 array="images",
                 transform=None):

        self.transform = transform

        os.makedirs(data_dir, exist_ok=False)
        self.file_path = os.path.join(data_shape,filename)
        assert not os.path.exists(self.file_path), "File already exists: "+self.file_path

        self.file = tables.open_file(self.file_path,mode='w')
        data=self.file.create_group(group)
        atom = tables.Atom.from_dtype(np.dtype("float32"))

        self.array = self.file.create_array(data,array,
                                             atom=atom,
                                             shape=(database_size,*data_shape))

    def save_batch(self, batch, first_index=0):
        if isinstance(batch,np.ndarray):
            npbatch = batch
        else:
            npbatch = batch.numpy()
        self.array[first_index:first_index+batch.shape[0]] = npbatch

    def finalize(self):
        self.file.close()


def generate_dcgan32_inversion_dataset(dataset_root="data/processed/dcgan32_inversion",
                                       dataset_size=100000,
                                       batch_size=128,
                                       generator_checkpoint_path="models/dcgan32v1/model_weights/checkpointG.2020_04_26",
                                       device=torch.device("cuda:0"),
                                       torch_seed = None,
                                       *,
                                       image_saver,
                                       label_saver):
    """General procedure for generating pairs of z,G(z) from the generator of DCGAN32
    Needs an image_saver and a label_saver object, implemented in different versions above
    """

    generator = DCGen32().to(device)
    generator_checkpoint = torch.load(generator_checkpoint_path)
    generator.load_state_dict(generator_checkpoint["model_state_dict"])
    generator.eval()
    print("Successfully initialized generator")

    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    n_images = 0
    while dataset_size > n_images:
        x = torch.zeros(batch_size, 100).normal_(0, 1.).to(device)
        batch = generator(x).cpu()
        x = x.cpu()

        image_saver.save_batch(batch,first_index=n_images)
        label_saver.save_batch(x,first_index=n_images)
        n_images += batch_size
        print("Generated and saved: {}/{} | {}".format(n_images,dataset_size,"="*(10*n_images//dataset_size)))

    if n_images < dataset_size:
        x = torch.zeros(dataset_size-n_images, 100).normal_(0, 1.).to(device)
        batch = generator(x).cpu()
        x = x.cpu()

        image_saver.save_batch(batch,first_index=n_images)
        label_saver.save_batch(x,first_index=n_images)

    image_saver.finalize()
    label_saver.finalize()


def generate_dcgan32_inversion_dataset_many_files(dataset_root="data/processed/dcgan32_inversion",
                                       dataset_size=100000,
                                       batch_size=128,
                                       generator_checkpoint_path="models/dcgan32v1/model_weights/checkpointG.2020_04_26",
                                       device=torch.device("cuda:0"),
                                       torch_seed = None):

    os.makedirs(dataset_root, exist_ok=True)

    # Preparing the image saver
    image_dir = os.path.join(dataset_root,"imgs")
    imageconverter = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        torchvision.transforms.ToPILImage()
    ])
    image_saver = ImagePerFileBatchSaver(image_dir=image_dir,transform=imageconverter)

    # Preparing the label saver
    label_dir = os.path.join(dataset_root,"vecs")
    label_saver = LabelPerTorchFileBatchSaver(label_dir=label_dir)

    # Run the actual generator

    generate_dcgan32_inversion_dataset(dataset_root=dataset_root,
                                       dataset_size=dataset_size,
                                       batch_size=batch_size,
                                       generator_checkpoint_path=generator_checkpoint_path,
                                       device=device,
                                       torch_seed=torch_seed,
                                       image_saver=image_saver,
                                       label_saver=label_saver)


def generate_dcgan32_inversion_dataset_many_images_one_labelfile(dataset_root="data/processed/dcgan32_inversion",
                                       dataset_size=100000,
                                       batch_size=128,
                                       generator_checkpoint_path="models/dcgan32v1/model_weights/checkpointG.2020_04_26",
                                       device=torch.device("cuda:0"),
                                       torch_seed = None):

    os.makedirs(dataset_root, exist_ok=True)

    # Preparing the image saver
    image_dir = os.path.join(dataset_root,"imgs")
    imageconverter = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        torchvision.transforms.ToPILImage()
    ])
    image_saver = ImagePerFileBatchSaver(image_dir=image_dir,transform=imageconverter)

    # Preparing the label saver
    label_dir = os.path.join(dataset_root,"vecs")
    label_saver = LabelOneTorchFileBatchSaver(label_dir=label_dir,
                                              database_size=dataset_size,
                                              label_shape=(100,),
                                              device=device)

    # Run the actual generator
    generate_dcgan32_inversion_dataset(dataset_root=dataset_root,
                                       dataset_size=dataset_size,
                                       batch_size=batch_size,
                                       generator_checkpoint_path=generator_checkpoint_path,
                                       device=device,
                                       torch_seed=torch_seed,
                                       image_saver=image_saver,
                                       label_saver=label_saver)


def generate_dcgan32_inversion_dataset_two_h5(dataset_root="data/processed/dcgan32_inversion",
                                       dataset_size=100000,
                                       batch_size=128,
                                       generator_checkpoint_path="models/dcgan32v1/model_weights/checkpointG.2020_04_26",
                                       device=torch.device("cuda:0"),
                                       torch_seed = None):

    os.makedirs(dataset_root, exist_ok=True)

    # Preparing the image saver
    image_saver = HDF5TensorSaver(data_dir=dataset_root,
                                  filename="images.h5",
                                  database_size=dataset_size,
                                  data_shape=(32,32,3))

    # Preparing the label saver
    label_saver = HDF5TensorSaver(data_dir=dataset_root,
                                  filename="labels.h5",
                                  database_size=dataset_size,
                                  data_shape=(100,))


    # Run the actual generator
    generate_dcgan32_inversion_dataset(dataset_root=dataset_root,
                                       dataset_size=dataset_size,
                                       batch_size=batch_size,
                                       generator_checkpoint_path=generator_checkpoint_path,
                                       device=device,
                                       torch_seed=torch_seed,
                                       image_saver=image_saver,
                                       label_saver=label_saver)