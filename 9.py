import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the image
img_path = "/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignemnt 2/Assignment/Crop_field_cropped.jpg"
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check with os.path.exists()"

edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)
x = indices[1].reshape(-1, 1)  # Reshape to column vector
y = indices[0]

lr = linear_model.LinearRegression()
lr.fit(x, y)

ransac = linear_model.RANSACRegressor()
ransac.fit(x, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(x.min(), x.max()).reshape(-1, 1)  # Reshape to column vector
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

print("Estimated coefficients (true, linear regression, RANSAC):")
print(lr.coef_, ransac.estimator_.coef_)

plt.scatter(
    x[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    x[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
)
plt.plot(line_X, line_y, color="navy", linewidth=2, label="Linear regressor")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=2,
    label="RANSAC regressor",
)
plt.legend(loc="lower right")
plt.xlabel("Input")
plt.ylabel("Response")
plt.gca().invert_yaxis() 
plt.savefig('answer9.png')
plt.show()