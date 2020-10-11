from dataset.HO_Data.data_util import *
from dataset.HO_Data.vis_util import *

class V2VVoxelization(object):
    def __init__(self, cubic_size, augmentation=False):
        self.cubic_size = cubic_size
        self.cropped_size1, self.original_size1 = 88,96
        self.cropped_size2, self.original_size2 = 44,48
        self.sizes1 = (self.cubic_size, self.cropped_size1, self.original_size1)
        self.sizes2 = (self.cubic_size, self.cropped_size2, self.original_size2)
        self.pool_factor = 2
        self.std = 1.7
        self.augmentation = augmentation
        self.extract_coord_from_output = extract_coord_from_output
        output_size = int(self.cropped_size1 / self.pool_factor)
        # Note, range(size) and indexing = 'ij'
        self.d3outputs = np.meshgrid(np.arange(output_size), np.arange(output_size), np.arange(output_size),
                                     indexing='ij')

    def voxelize88(self, points, refpoint):
        new_size, angle, trans = 100, 0, self.original_size1 / 2 - self.cropped_size1 / 2
        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes1)
        return input.reshape((1, *input.shape))

    def voxelize44(self, points, refpoint):
        new_size, angle, trans = 100, 0, self.original_size2 / 2 - self.cropped_size2 / 2
        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes2)
        return input.reshape((1, *input.shape))


