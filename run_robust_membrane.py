from Modules.robust_membrane_filter import RobustMembraneFilter
import matplotlib.pyplot as plt
from PIL import Image

alpha = 100
img = Image.open('guanajuato.jpg')
model = RobustMembraneFilter(
    img,
    alpha
)
output = model.run(300)
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)
ax1.imshow(img)
ax1.axis("off")
ax2.imshow(output)
ax2.axis("off")
plt.tight_layout()
plt.show()
