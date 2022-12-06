from torchvision.transforms import (
    ToPILImage,
    ToTensor,
)
from torch.nn.functional import pad
from torch.optim import Adam
from PIL.Image import Image
from torch import Tensor
from torch import mean
from torch import cuda
from torch.nn import (
    Parameter,
    MSELoss,
)


class QuadraticMembraneFilter:
    """
    -
    """

    def __init__(self,
                 image: Image,
                 alpha: int) -> None:
        self.to_image = ToPILImage()
        self.to_tensor = ToTensor()
        self.optimizer = None
        self.loss = MSELoss()
        self.output = None
        self.device = None
        self.image = None
        self.alpha = alpha
        self._get_device()
        self._get_tensors(image)
        self._get_optimizer()

    def _get_tensors(self,
                     image: Image) -> None:
        """
        -
        """
        self.image = self.to_tensor(image)
        self.image = self._to_device(self.image)
        self.output = self.image.clone()
        self.output = self._to_device(self.output)
        self.output = self.output.requires_grad_(True)
        self.output = Parameter(self.output)

    def _get_device(self) -> None:
        """
        -
        """
        if cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def _to_device(self,
                   tensor: Tensor) -> None:
        """
        -
        """
        tensor = tensor.to(self.device)
        return tensor

    def _get_optimizer(self,) -> None:
        """
        -
        """
        self.optimizer = Adam([self.output])

    def _L2_gradient(self, f):
        '''
        Calcula el promedio de la magnitud del gradiente espacial
        de la imagen multicanal f  (c,h,w)
        '''
        # gradiente
        fx, fy = self._gradient(f)
        # promedio de la magnitud del gradiente
        l2 = mean(fx**2+fy**2)
        return l2

    def _gradient(self, f) -> tuple:
        '''
        Entrada
        f:      (c,h,w), float32 or float64
        Resultados
        fx, fy: (c,h,w)
        '''
        # corrimientos
        # pad last dim by (0, 1)
        f_10 = pad(f, (0, 1, 0, 0))
        f_10 = f_10[:, :, 1:]
        # pad 2nd to last dim by (0, 1)
        f_01 = pad(f, (0, 0, 0, 1))
        f_01 = f_01[:, 1:, :]
        # primeras diferencias adelantadas
        fx = f_10 - f
        fy = f_01 - f
        # derivadas en la frontera
        # fx will have zeros in the last column
        fx[:, :, -1] = 0
        # fy will have zeros in the last row
        fy[:, -1, :] = 0

        return fx, fy

    def run(self,
            epochs: int) -> tuple:
        """
        -
        """
        self._fit(epochs)
        image = self.to_image(self.image)
        output = self.to_image(self.output)
        return image, output

    def _fit(self,
             epochs: int) -> None:
        """
        -
        """
        for _ in range(epochs):
            # Inicializa gradiente
            self.optimizer.zero_grad()
            # Término de datos
            loss_d = self.loss(
                self.output,
                self.image
            )
            # Término de regularización
            loss_r = self._L2_gradient(self.output)
            # Costo total
            loss = loss_d + self.alpha * loss_r
            # Retropropagación (gradiente)
            loss.backward()
            # Actualiza los parametros del modelo
            self.optimizer.step()
