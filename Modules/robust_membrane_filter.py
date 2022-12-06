from .quadratic_membrane_filter import QuadraticMembraneFilter
from PIL.Image import Image
from torch import Tensor
from torch import (
    mean,
    abs,
)


class RobustMembraneFilter(QuadraticMembraneFilter):
    """
    -
    """

    def __init__(self,
                 image: Image,
                 alpha: int) -> None:
        super().__init__(
            image,
            alpha,
        )

    def _L1_gradient(self,
                     f: Tensor) -> Tensor:
        '''
        Calcula el promedio de la magnitud del gradiente espacial
        de la imagen multicanal f  (c,h,w)
        '''
        # gradiente
        fx, fy = self._gradient(f)
        # promedio de la magnitud del gradiente
        l1 = mean(abs(fx) + abs(fy))
        return l1

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
            loss_r = self._L1_gradient(self.output)
            # Costo total
            loss = loss_d + self.alpha * loss_r
            # Retropropagación (gradiente)
            loss.backward()
            # Actualiza los parametros del modelo
            self.optimizer.step()
