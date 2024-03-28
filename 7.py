import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load the image and perform edge detection
img_path = "/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignemnt 2/Assignment/Crop_field_cropped.jpg"
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check with os.path.exists()"
edges = cv.Canny(img, 550, 690)

# Find edge points
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

# Calculate least squares fit
X = np.vstack([x, np.ones(len(x))]).T
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
m = beta_hat[0]
c = beta_hat[1]

# Add noise to y values
np.random.seed(45)
noise = np.random.randn(len(x))
y_noisy = y + noise

# Total Least Squares (TLS) line
u11 = np.sum((x - np.mean(x))**2)
u12 = np.sum((x - np.mean(x))*(y_noisy - np.mean(y_noisy)))
u21 = u12
u22 = np.sum((y_noisy - np.mean(y_noisy))**2)
U = np.array([[u11, u12], [u21, u22]])
w, v = np.linalg.eig(U)
smallest_eigenvector = v[:, np.argmin(w)]
a = smallest_eigenvector[0]
b = smallest_eigenvector[1]
d = a*np.mean(x) + b*np.mean(y_noisy)
mstar = -a/b
cstar = d/b

# Determine the range of x-values
x_min = np.min(x)
x_max = np.max(x)

# Calculate corresponding y-values for the lines
y_ls_fit = m * np.array([x_min, x_max]) + c
y_tls_fit = mstar * np.array([x_min, x_max]) + cstar

angle_radians = np.arctan(mstar)
angle_degrees = np.degrees(angle_radians)

print("Estimated angle of the crop field based on total least-squares-fit:", angle_degrees, "degrees")
