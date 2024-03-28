import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import linear_model

img_path = "/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignemnt 2/Assignment/Crop_field_cropped.jpg"
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check with os.path.exists()"

edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)
x = indices[1].reshape(-1, 1)  # Reshape to column vector
y = indices[0]

ransac = linear_model.RANSACRegressor()
ransac.fit(x, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(x.min(), x.max()).reshape(-1, 1)  # Reshape to column vector
line_y_ransac = ransac.predict(line_X)

slope_ransac = ransac.estimator_.coef_[0]

crop_field_angle = np.degrees(np.arctan(slope_ransac))
print("Estimated angle of the crop field:", crop_field_angle, "degrees")

plt.scatter(
    x[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    x[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
)
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=2,
    label="RANSAC regressor",
)
plt.legend(loc="lower right")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Estimated Line Using RANSAC Algorithm")
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
plt.savefig('answer10')
plt.show()
