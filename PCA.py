import numpy as np
import matplotlib.pyplot as plt

class1 = np.array([1,3,4,5,7,6,9,10,12,13])
class2 = np.array([2,20,3,6,5,23,4,1,12,1])
all_data = np.vstack((class1, class2))
all_data = all_data.T
print(all_data)
mean1 = class1.mean()
mean2 = class2.mean()

mui = all_data - np.array([mean1, mean2])
print(f"mui shape : {mui.shape} ,\n mui : {mui}")
# mui = np.array([[mean1], [mean2]]) - all_data
cov = (1/(all_data.shape[0]-1)) * np.dot(mui.T, mui)
# cov = np.cov(mui)
# cov = np.cov(all_data.T)
print(cov)
print(mui)
# calculating eigen values and eigen vectors
eigen_Values, eigen_Vector = np.linalg.eig(cov)
print(eigen_Values)
print(eigen_Vector)
sorted_eigen_pairs = []
for i in range(len(eigen_Values)):
    sorted_eigen_pairs.append([eigen_Values[i], eigen_Vector[i]])
sorted_eigen_pairs = sorted(sorted_eigen_pairs, reverse=True)
print(sorted_eigen_pairs)

a, b = sorted_eigen_pairs[0][1][0], sorted_eigen_pairs[0][1][1]
a1, b1 = sorted_eigen_pairs[1][1][0], sorted_eigen_pairs[1][1][1]
# x2 - x1 / a = y2 - y1/ b= -(ax1 + by1 + c) / (a2 + b2)
# final_points = []
# for j in range(all_data.shape[1]):
#     points = all_data[:,j]    
#     x2 = (-(a * points[0] + b * points[1]) / (a**2 + b**2)) * a + points[0]
#     y2 = (-(a * points[0] + b * points[1]) / (a**2 + b**2)) * b + points[1]
#     final_points.append([x2,y2])
# print(final_points)

# final_points = np.array(final_points)
sorted_eigen_pairs[0][1].reshape(-1,1).shape
eigen_Vector
all_data.shape
final_points = np.dot(mui, eigen_Vector)

final_points[:,0]
final_points
final_points.shape
before_points = np.dot(final_points,eigen_Vector.T)
before_points
x = np.arange(-20,50)
y = (b/a) * x
y1 = (b1/a1) * x
plt.scatter(final_points[:,0], final_points[:,1], color = 'red')
plt.scatter(class1,class2, color = 'green')
# plt.plot([final_points[:,0],class1], [final_points[:,1],class2])
# plt.scatter(mui[0,:], mui[1,:])
# plt.plot([final_points[:,0],mui[0, :]], [final_points[:,1],mui[1, :]])
plt.plot(x, y)
plt.plot(x,y1)
plt.show()

