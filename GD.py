import numpy as np

# 随机写的样本数据
x = np.array([[2.3, 3.5], [4.8, 7.6], [3.2, 7.7], [18.5, 32], [33.2, 17],
              [7, 9]])
# 样本点对应的输出
y = np.array([15.88, 20.69, 34.22, 25, 49.34, 20])

# 归一化，不做归一化可能会导致无法收敛或者收敛速度变慢
max_x = x.max(axis=0)
x = x / max_x

# 参数第一位是截距，后面两位分别是x的对应权重
theta = np.array([1, 1, 1])
# 更新步长
alpha = 0.01
# 停止迭代的阈值
threshold = 0.00001


# 批量梯度下降
def get_grad_BGD(theta, x, y):
  grad = [0, 0, 0]
  comm = theta[0] + theta[1] * x[:, 0] + theta[2] * x[:, 1] - y
  grad[0] = 2. * np.mean(comm)
  grad[1] = 2. * np.mean(x[:, 0] * comm)
  grad[2] = 2. * np.mean(x[:, 1] * comm)
  return np.array(grad)


# 随机梯度下降
def get_grad_SGD(theta, x, y):
  grad = [0, 0, 0]
  # 随机取一个样本和y值
  r = np.random.randint(0, len(x))
  xi = x[r]
  yi = y[r]
  comm = theta[0] + theta[1] * xi[0] + theta[2] * xi[1] - yi
  grad[0] = 2. * comm
  grad[1] = 2. * xi[0] * comm
  grad[2] = 2. * xi[1] * comm
  return np.array(grad)


# 小批量梯度下降
def get_grad_MBGD(theta, batch, x, y):
  grad = [0, 0, 0]
  # 随机取一个样本和y值
  r = np.random.choice(len(x), batch, replace=False)
  xm = x[r]
  ym = y[r]
  comm = theta[0] + theta[1] * xm[:, 0] + theta[2] * xm[:, 1] - ym
  grad[0] = 2. * np.mean(comm)
  grad[1] = 2. * np.mean(xm[:, 0] * comm)
  grad[2] = 2. * np.mean(xm[:, 1] * comm)
  return np.array(grad)


batch = 4


# 更新参数theta
def update_theta(theta, alpha, grad):
  new_theta = theta - alpha * grad
  return new_theta


# 计算RMSE,即我们当前损失值的平方根
def get_rmse(theta, x, y):
  squared_err = (theta[0] + theta[1] * x[:, 0] + theta[2] * x[:, 1] - y)**2
  rmse = np.sqrt(np.mean(squared_err))
  return rmse


# 计算梯度的模
def grad_norm(grad):
  g = np.array(grad)
  norm = np.sqrt(np.sum(g**2))
  return norm


# 进行一次计算，得到新旧梯度的模以及新的theta值
def iterate(alpha, theta, x, y):
  grad = get_grad_BGD(theta, x, y)
  # grad = get_grad_MBGD(theta, batch, x, y)
  norm = grad_norm(grad)
  theta = update_theta(theta, alpha, grad)
  # grad = get_grad_MBGD(theta, batch, x, y)
  grad = get_grad_BGD(theta, x, y)
  new_norm = grad_norm(grad)
  return norm, new_norm, theta


# 设置迭代初始值
count = 1
new_norm = 1
norm = 0
while np.abs(new_norm - norm) > threshold:
  norm, new_norm, theta = iterate(alpha, theta, x, y)
  count += 1

print('iteration count: %s' % count)
print('Coef: [%s %s] ' % (theta[1], theta[2]))
print('Intercept %s' % theta[0])
rms = get_rmse(theta, x, y)
print('RMSE: %s' % rms)

# 通过sklearn的线性回归工具来拟合并且计算rmse，和我们自己实现的梯度下降得到的rmse做对比
print('--------------------------')
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)
print('Sklearn Coef: %s' % lr.coef_)
print('Sklearn Intercept: %s' % lr.intercept_)
sktheta = [lr.intercept_, lr.coef_[0], lr.coef_[1]]
rms = get_rmse(sktheta, x, y)
print('Sklearn RMSE: %s' % rms)