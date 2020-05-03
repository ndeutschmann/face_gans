"""Datasets based on accesing indexable list-like objects"""
import tables


class MultiIndexableDataset:
    def __init__(self,arrays,transforms=None):
        self.arrays = arrays
        self.transforms = transforms

    def __getitem__(self, item):
        if self.transforms is None:
            return tuple([array[item] for array in self.arrays])

        return tuple(array[item] if self.transforms[item] is None
                      else self.transforms[item](array[item])
                      for array in self.arrays)


class MultiHDF5TablesDataset(MultiIndexableDataset):
    def __init__(self, files, array_nodes, load_in_memory=None, transforms=None):
        assert len(files) == len(array_nodes)

        if load_in_memory is None:
            self.mem = (False,)*len(files)
        else:
            self.mem = load_in_memory
        assert len(self.mem) == len(files)

        self.files = []
        arrays = []
        for i,file in enumerate(files):
            newfile = tables.open_file(file,mode="r")
            self.files.append(newfile)
            newarray = newfile.get_node(array_nodes[i])
            assert isinstance(newarray,tables.array.Array), \
                "Node {} in file {} is not an array".format(array_nodes[i],file)
            arrays.append(newarray)

        assert len(set(len(a) for a in arrays)) == 1, "All arrays must be of the same length"

        super(MultiHDF5TablesDataset, self).__init__(arrays, transforms=transforms)

    def __len__(self):
        return len(self.arrays[0])

    def close_files(self):
        for file in self.files:
            file.close()
