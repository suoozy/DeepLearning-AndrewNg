"""
You will build a logistic regression classifier to recognize cats.

tips：
1.You have to check if there is possibly overfitting.
  It happens when the training accuracy is a lot higher than the test accuracy.
2.In deep learning, we usually recommend that you:
   Choose the learning rate that better minimizes the cost function.
   If your model overfits, use other techniques to reduce overfitting.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import imageio
import skimage
from PIL import Image
from lr_utils import load_dataset  #一个加载资料包里面的数据的简单功能的库

# =========================Loading the data (cat/non-cat)=============================
# train_set_x_orig ：保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
# train_set_y ：保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
# test_set_x_orig ：保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
# test_set_y ： 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
# classes ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# =================================Example of a picture=============================
index = 25
plt.imshow(train_set_x_orig[index])
plt.show()
# plt.imshow()函数负责对图像进行处理，并显示其格式
# plt.show()则是将plt.imshow()处理后的函数显示出来
print ("y = " + str(train_set_y[:,index]) + ", it's a '"
        + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")
# np.squeeze是为了压缩维度，只有压缩后的值才能进行解码

# ====================Figure out the dimensions and shapes of the problem================
m_train = train_set_y.shape[1]   # number of training examples
m_test = test_set_y.shape[1]     # number of test examples
num_px = train_set_x_orig.shape[1]      # height = width of a training image
# Remember that train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3).

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# ==========================Reshape the training and test examples=======================
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#==============="Standardize" the data（将数据转换到0-1之间，类似于正态分布标准化）========
# 因为像素都在0-255之间，所以除以255就可以让数据集的值处于0-1之间
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

#===================================几个自定义函数======================================
def sigmoid(z):

    s = 1/(1+np.exp(-z))

    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros(shape=(dim, 1))
    b = 0

    # assert是断言语句，用来确保维度正确
    # 如果输入的布尔值为False，则报错并终止程序，避免更大的错误
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(X,Y,w,b):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation

    """

    m = X.shape[1]    # number of training examples

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)    # np.squeeze是为了压缩维度，只有压缩后的值才能进行解码
    assert (cost.shape == ())

    # 创建一个字典，把dw和db保存起来。
    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []  # 定义代价的空列表

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads,cost = propagate(X,Y,w,b)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w -= learning_rate*dw    #need to broadcast
        b -= learning_rate*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            # append() 方法用于在列表末尾添加新的对象。该方法无返回值，但是会修改原来的列表
            # 该方法可以用于在循环迭代过程中保存每一次的运行结果

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X

    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):   # A.shape[1]=样本数m
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        if A[0,i]>0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
        # 可以简写为：Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction

#==============================Merge all functions into a model=========================
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###
    # initialize parameters with zeros
    w,b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    params,grads,costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = params["w"]
    b = params["b"]

    # Predict test/train set examples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#====================Example of a picture that was wrongly classified=====================
index = 5
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
plt.show()
print ("y = " + str(test_set_y[0, index])+", you predicted that it is a \""
       + classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") +  "\" picture.")


#=============================Plot learning curve (with costs)==========================
costs = np.squeeze(d['costs'])        # 将数组转换为秩为1的数组
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

#=====================================多个不同学习速率的比较==============================
learning_rates = [0.01, 0.001, 0.0001]
models = {}      #为什么？？？？？？？
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
# plt.legend(loc=图例中所有figure的位置，shadow=是否在图例后面画一个阴影)
frame = legend.get_frame()  # 返回legend所在的方形对象
frame.set_facecolor('0.90')   # 设置图例legend背景透明度
plt.show()

#=================================Test with your own image================================
# PUT YOUR IMAGE NAME
my_image = "my_image.jpg"

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(imageio.imread(fname))
my_image = skimage.transform.resize(image, output_shape=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) +
      ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image)),].decode("utf-8")
      +  "\" picture.")