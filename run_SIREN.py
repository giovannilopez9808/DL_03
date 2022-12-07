from Modules.SIREN import SIRENModel
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('Cameraman.tif')
params = {
    "image": img,
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
    }
}
schedule = SIRENModel(
    params
)
schedule.train()
y = schedule.test()
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(10, 5)
)
ax1.imshow(y, 'gray')
ax1.axis('off')
ax2.imshow(img, 'gray')
ax2.axis('off')
plt.show()
