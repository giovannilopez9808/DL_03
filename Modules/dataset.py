from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import Tensor
from numpy import (
    meshgrid,
    arange,
    stack,
    array,
)


class Dataset:
    """
    -
    """

    def __init__(self,
                 params: dict) -> None:
        self.image = params["image"]
        self.to_tensor = ToTensor()
        self.image_tensor = None
        self.batch_size = None
        self.data = None
        self._get_dataset()

    def _get_dataset(self) -> Tensor:
        x = self.create_columns()
        y = self._get_target()
        self.data = DataLoader(
            dataset=[*zip(x, y)],
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_columns(self) -> Tensor:
        rows, cols = self.image.size
        self.batch_size = rows*(cols//10)
        i, j = meshgrid(
            arange(rows),
            arange(cols),
        )
        i = self._normalization(
            i,
            rows
        )
        j = self._normalization(
            j, cols
        )
        x = stack([i, j])
        x = x.T
        x = self.to_tensor(x)
        x = x[0]
        return x

    def _get_target(self) -> Tensor:
        self.image_tensor = self.to_tensor(
            self.image
        )
        y = self.image_tensor.flatten()
        y = y-0.5
        return y

    def _normalization(self,
                       matrix: array,
                       size: int) -> array:
        vector = matrix.flatten()
        vector = vector.astype("float32")
        vector = (vector/size)-0.5
        return vector
