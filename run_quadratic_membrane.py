from Modules.quadratic_membrane_filter import QuadraticMembraneFilter
import matplotlib.pyplot as plt
from PIL import Image

alpha = 100
img = Image.open('guanajuato.jpg')
model = QuadraticMembraneFilter(
    img,
    alpha
)
im_g, im_f = model.run(300)
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)
ax1.imshow(im_g)
ax1.axis("off")
ax2.imshow(im_f)
ax2.axis("off")
plt.tight_layout()
plt.show()
