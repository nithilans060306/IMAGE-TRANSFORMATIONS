# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the necessary libraries and read the original image and save it as a image variable.

### Step2:
Translate the image using a function warpPerpective()

### Step3:
Scale the image by multiplying the rows and columns with a float value.

### Step4:
Shear the image in both the rows and columns.

### Step5:
Find the reflection of the image.

### step 6:
Rotate the image using angle function.

## Program:
```python
Developed By:
Register Number:

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Chennai_Central.jpg')
image.shape

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

i)Image Translation
# i) Image Translation
tx, ty = 100, 200  # Translation factors (shift by 100 pixels horizontally and 50 vertically)
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix: 
# [1, 0, tx] - Horizontal shift by tx
# [0, 1, ty] - Vertical shift by ty
translated_image = cv2.warpAffine(image, M_translation, (636, 438))

plt.imshow(translated_image[:,:,::-1])
plt.title("Translated Image")
plt.axis('on')
plt.show()

ii) Image Scaling
# Image Scaling
fx, fy = 2.0, 1.0  
scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

plt.imshow(scaled_image[:,:,::-1]) 
plt.title("Scaled Image") 
plt.axis('on')
plt.show()

iii)Image shearing
# Image Shearing
shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  
sheared_image = cv2.warpAffine(image, shear_matrix, (636, 438))

plt.imshow(sheared_image[:,:,::-1])
plt.title("Sheared Image") 
plt.axis('on')
plt.show()

iv)Image Reflection
# Image Reflection
reflected_image = cv2.flip(image, 2)  # Flip the image horizontally (1 means horizontal flip)

# flip: 1 means horizontal flip, 0 would be vertical flip, -1 would flip both axes

# Show original image 
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis('off')

# Show reflected image 
plt.subplot(1, 2, 2)
plt.imshow(reflected_image[:,:,::-1])
plt.title("Reflected Image")
plt.axis('off')

plt.tight_layout()
plt.show()

v)Image Rotation
# Image Rotation
(height, width) = image.shape[:2]  # Get the image height and width
angle = 45  # Rotation angle in degrees (rotate by 45 degrees)
center = (width // 2, height // 2)  # Set the center of rotation to the image center
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)  # Get the rotation matrix
# getRotationMatrix2D: Takes the center of rotation, angle, and scale factor (1 means no scaling)
rotated_image = cv2.warpAffine(image, M_rotation, (width, height))  # Apply rotation

plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))  # Display the rotated image
plt.title("Rotated Image")  # Set title
plt.axis('off')

# Image Rotation
image.shape
angle = 145  
center = (636 // 2, 438 // 2)  
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)  
# getRotationMatrix2D: Takes the center of rotation, angle, and scale factor (1 means no scaling)
rotated_image = cv2.warpAffine(image, M_rotation, (width, height))

plt.imshow(rotated_image[:,:,::-1])  # Display the rotated image
plt.title("Rotated Image")  # Set title
plt.axis('off')
plt.show()

vi)Image Cropping
# Image Cropping
x, y, w, h = 0, 0, 200, 150  

cropped_image = image[y:y+h, x:x+w]   # Format: image[start_row:end_row, start_col:end_col]

# Show original image 
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis('on')

# Show reflected image 
plt.subplot(1, 2, 2)
plt.imshow(cropped_image[:,:,::-1])
plt.title("Cropped Image")
plt.axis('on')

plt.tight_layout()
plt.show()

```
## Output:
### i)Image Translation
<img width="690" height="508" alt="image" src="https://github.com/user-attachments/assets/dc1c45ff-3262-41bc-af01-98c6a2c016d2" />

### ii) Image Scaling
<img width="688" height="295" alt="image" src="https://github.com/user-attachments/assets/9d220109-b6aa-429f-a4a2-5e755dd09e62" />

### iii)Image shearing
<img width="685" height="506" alt="image" src="https://github.com/user-attachments/assets/8b480b1d-d2d7-489b-b5d5-204672a3c85d" />

### iv)Image Reflection
<img width="874" height="328" alt="image" src="https://github.com/user-attachments/assets/a2854d34-49a5-4c45-aa5a-435599f4c269" />

### v)Image Rotation
<img width="641" height="479" alt="image" src="https://github.com/user-attachments/assets/d4d50e4c-b731-47f9-a399-c6ac4cb14fcc" />

<img width="646" height="479" alt="image" src="https://github.com/user-attachments/assets/260a2e80-e8ee-4ad2-b082-1dba236c58c5" />

### vi)Image Cropping
<img width="875" height="353" alt="image" src="https://github.com/user-attachments/assets/424e26be-2002-4f9b-bcfd-a39a899d5c82" />

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
