import os
from torch import load
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader,is_image_file


class ImageTargetFoldersDataset(VisionDataset):
    """Dataset with a folder for images, and a folder for targets
    This is for targets stored in individual files, which is overall a bad idea

    Image file names must match target file names up to the extension
    """

    def __init__(self,
                 root=None,
                 image_dir=None,
                 target_dir=None,
                 target_ext="pkl",
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 target_loader=load):

        super(ImageTargetFoldersDataset, self).__init__(root=root,
                                                        transforms=None,
                                                        transform=transform,
                                                        target_transform=target_transform)

        self.loader = loader
        self.target_loader = target_loader

        # By default the structure is the following
        #   root/
        #   ├── imgs
        #   └── tgts
        # if either image_dir or target_dir is None then we use this structure
        # Otherwise, we look for the following structure
        #   root/
        #   ├── $image_dir
        #   └── $target_dir
        # Finally, if root is None, we expect two independent paths for the images and targets

        if root is None:
            self.image_dir = os.path.abspath(image_dir)
            self.target_dir = os.path.abspath(target_dir)
        else:
            if image_dir is None:
                self.image_dir = os.path.abspath(os.path.join(root, "imgs"))
            else:
                self.image_dir = os.path.abspath(os.path.join(root, image_dir))
            if target_dir is None:
                self.target_dir = os.path.abspath(os.path.join(root, "tgts"))
            else:
                self.target_dir = os.path.abspath(os.path.join(root, target_dir))

        assert os.path.isdir(self.image_dir), "This is not a directory:\n" + str(self.image_dir)
        assert os.path.isdir(self.target_dir), "This is not a directory:\n" + str(self.target_dir)

        images = []
        targets = []

        for im in os.listdir(self.image_dir):
            impath = os.path.join(self.image_dir, im)
            if os.path.isfile(impath) and is_image_file(impath):
                # Found an image, let's check that there is a matching target
                imname = os.path.splitext(im)[0]
                tpath = os.path.join(self.target_dir, imname + "." + target_ext)
                if os.path.isfile(tpath):
                    images.append(impath)
                    targets.append(tpath)

        self.images = images
        self.targets = targets

    def __getitem__(self, item):
        impath = self.images[item]
        tpath = self.targets[item]

        image = self.loader(impath)
        target = self.target_loader(tpath)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)



