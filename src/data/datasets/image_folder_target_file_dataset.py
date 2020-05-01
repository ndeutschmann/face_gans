import os
from torch import load
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader,is_image_file


class ImageFoldersTargetFileDataset(VisionDataset):
    """Dataset with a folder for images, and a file for the labels
    Images are supposed to be X.FMT where X is an integer and FMT is an image format
    Targets are store in a single file which can be loaded as an indexable object Obj such that
    Obj[X] is the label of image X.
    """

    def __init__(self,
                 root=None,
                 image_dir=None,
                 target_path=None,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 target_loader=load):

        super(ImageFoldersTargetFileDataset, self).__init__(root=root,
                                                        transforms=None,
                                                        transform=transform,
                                                        target_transform=target_transform)

        self.loader = loader
        self.target_loader = target_loader

        # By default the structure is the following
        #   root/
        #   ├── imgs
        #   └── tgts/tgts.pkl
        # if either image_dir or target_dir is None then we use this structure
        # Otherwise, we look for the following structure
        #   root/
        #   ├── $image_dir
        #   └── $target_path <- a path to a file
        # Finally, if root is None, we expect two independent paths for the images and targets

        if root is None:
            self.image_dir = os.path.abspath(image_dir)
            self.target_path = os.path.abspath(target_path)
        else:
            if image_dir is None:
                self.image_dir = os.path.abspath(os.path.join(root, "imgs"))
            else:
                self.image_dir = os.path.abspath(os.path.join(root, image_dir))
            if target_path is None:
                self.target_path = os.path.abspath(os.path.join(root, "tgts","tgts.pkl"))
            else:
                self.target_path = os.path.abspath(os.path.join(root, target_path))

        assert os.path.isdir(self.image_dir), "This is not a directory:\n" + str(self.image_dir)
        assert os.path.isfile(self.target_path), "This is not a file:\n" + str(self.target_path)

        images = []

        self.targets = target_loader(self.target_path)
        target_accessed = [False for _ in range(len(self.targets))]

        for im in os.listdir(self.image_dir):
            impath = os.path.join(self.image_dir, im)
            if os.path.isfile(impath) and is_image_file(impath):
                # Found an image, let's check that there is a matching target
                im_index = int(os.path.splitext(im)[0])
                target = self.targets[im_index]
                target_accessed[im_index] = True
                images.append(impath)

        assert all(target_accessed), "Image numbering incorrect. Please name your images from 0 to N without gaps."

        self.images = images

    def __getitem__(self, item):
        impath = self.images[item]

        image = self.loader(impath)
        target = self.targets[item]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)



