from .dataset import Dataset
from torch.optim import Adam
from torch.nn import (
    Sequential,
    MSELoss,
    Module,
    Linear
)
from numpy import (
    array,
    sqrt,
)
from torch import (
    no_grad,
    Tensor,
    cuda,
    save,
    sin,
)


class SineLayer(Module):
    """
    -
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 is_first: bool = False,
                 is_last: bool = False,
                 omega_0: int = 30) -> None:
        '''
        Implementa
        sin ( omega ( W x + b) )

        in_features   : (int) dimensión de entrada
        out_features  : (int) número de neuronas (dimensión de salida)
        bias          : -
        is_fisrt      : (boolean) se escala distinto la inicialización de la
        primera capa oculta y las restantes
        is_last       : (boolean) sn funcion de activacio en la capa de salida
        '''

        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last
        self.in_features = in_features
        self.linear = Linear(
            in_features,
            out_features,
            bias=bias
        )
        self.init_weights()

    def forward(self, x) -> Tensor:
        '''
        y = Phi(omega0 W (x+b) )
        '''
        out = self.linear(x)
        out = self.omega_0 * out
        out = sin(out)
        return out

    def forward_with_intermediate(self, x):
        '''
         z =  omega0  (W x + b)
         y =  sin(z)
         (y,z)
        '''
        # For visualization of activation distributions
        out = self.linear(x)
        out = self.omega_0*x
        out = sin(out), out if not self.last else out, out
        return out

    def init_weights(self):
        '''
        Inicialización escalada para considerar a la activación periodica
        Pretende mantener la respuesta de cada neurona dentro de una misma
        rama y evitar saltos entre ramas al entrenar los pesos
        '''
        with no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -sqrt(1 / self.in_features),
                    sqrt(1 / self.in_features)
                )
            else:
                self.linear.weight.uniform_(
                    -sqrt(6 / self.in_features) / self.omega_0,
                    sqrt(6 / self.in_features) / self.omega_0
                )


class SIRENnet(Module):
    """
    -
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int = 1) -> None:
        super().__init__()
        # Modelo inicialmente vacio
        self.net = []
        # Se agragan capas Seno ocultas
        is_first = True
        for i in range(len(hidden_dims)):
            is_last = False if i < len(hidden_dims)-1 else True
            self.net.append(
                SineLayer(
                    in_features=input_dim,
                    out_features=hidden_dims[i],
                    is_first=is_first,
                    is_last=is_last
                )
            )
            input_dim = hidden_dims[i]
            is_first = False
        self.net = Sequential(*self.net)

    def forward(self, x):
        out = self.net(x)
        return out

    def name(self):
        return "MLP"


class SIRENModel:
    def __init__(self,
                 params: dict) -> None:
        self.dataset = Dataset(params)
        self.model = SIRENnet(
            **params["SIREN"]
        )
        self.optimizer = None
        self.history = None
        self.device = None
        self.loss = None
        self._get_device()
        self._build()
        self.model = self._to_device(
            self.model
        )

    def _build(self) -> None:
        self.optimizer = Adam(self.model.parameters())
        self.loss = MSELoss()

    def train(self) -> None:
        epochs = 100
        history = list()
        for epoch in range(epochs):
            loss_epoch = 0
            for i, (x, y) in enumerate(self.dataset.data):
                x = self._to_device(x)
                y = self._to_device(y)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                y_pred = y_pred[:, 0]
                loss = self.loss(
                    y_pred,
                    y
                )
                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.data
            history.append(loss_epoch/i)
            print('Epoch: {:03}/{}  Loss: {:.6f}'.format(
                epoch,
                epochs,
                history[-1])
            )
        self.history = history
        save(
            self.model,
            "model_implicit.pt"
        )

    def test(self) -> None:
        rows, cols = self.dataset.image.size
        x = self.dataset.create_columns()
        x = self._to_device(x)
        y = self.model(x)
        y = self._to_cpu(y)
        y = y.reshape(rows, cols)
        return y

    def _to_cpu(self,
                tensor: Tensor) -> array:
        vector = tensor.detach().cpu().numpy()
        return vector

    def _to_device(self,
                   tensor: Tensor) -> Tensor:
        return tensor.to(self.device)

    def _get_device(self) -> None:
        if cuda.is_available():
            self.device = "cuda"
        else:
            self.device = 'cpu'
