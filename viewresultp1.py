import cv2
import numpy as np

image = cv2.imread('images/lion.jpeg')
masked_image = cv2.imread('outputs/mask_overlay.jpg')   # .jpg, not .jpeg
edited_image = cv2.imread('outputs/edited.jpg')          # .jpg, not .jpeg

# fail loudly instead of a confusing hstack crash if a path is wrong
for name, img in [('image', image), ('masked_image', masked_image), ('edited_image', edited_image)]:
    if img is None:
        raise FileNotFoundError(f"Could not load '{name}' — check the path exists")

cv2.imshow('edits', np.hstack((image, masked_image, edited_image)))
cv2.waitKey(0)          # keep window open until a key is pressed
cv2.destroyAllWindows()