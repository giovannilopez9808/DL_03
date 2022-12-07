from Modules.robust_membrane_filter import RobustMembraneFilter
from Modules.SIREN import SIRENModel
import matplotlib.pyplot as plt
from numpy import array
from PIL import Image


def plot(image: array,
         title: str,
         ax: plt.axes) -> None:
    ax.imshow(image,
              cmap="gray")
    ax.axis("off")
    ax.set_title(title)


img = Image.open('Cameraman.tif')
params = {
    "SIREN": {
        "input_dim": (2),
        "hidden_dims": [
            128,
            128,
            128,
            128,
            1
        ],
        "output_dim": 1,
    },
    "alpha": 100,
}
membrane_filter = RobustMembraneFilter(
    img,
    params["alpha"],
)
output = membrane_filter.run(300)
params["image"] = output
schedule = SIRENModel(
    params
)
schedule.train()
y = schedule.test()
fig, (ax1, ax2, ax3) = plt.subplots(
    1,
    3,
    figsize=(10, 5)
)
plot(
    img,
    "Orignal",
    ax1
)
plot(
    output,
    "Quadratic Membrane output",
    ax2
)
plot(
    y,
    "SIREN output",
    ax3
)
plt.tight_layout()
plt.show()
