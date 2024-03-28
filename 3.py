import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignemnt 2/Assignment/Crop_field_cropped.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img, 550, 690)
indices = np.where(edges != [0])
x = indices[1]
y = indices[0]

X = np.vstack([x, np.ones(len(x))]).T

# Calculate the least squares estimates
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Extract gradient and intercept
m = beta_hat[0]
c = beta_hat[1]

#np.random.seed(45)
#noise = np.random.randn(len(x))

plt.figure()
plt.scatter(x, y, color='red', s=1)
plt.title('Scatter plot of Edge Points')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.gca().invert_yaxis()  

plt.plot(x, m*x + c )
plt.legend([ 'Edge Points','Least-Squares Fit Line'])
#plt.savefig('answer3.png')
plt.show()

