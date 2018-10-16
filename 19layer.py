#加载VGG19模型并可视化一张图片前向传播的过程中每一层的输出 
#引入包 
#作者：Missayaa 
#来源：CSDN 
#原文：https://blog.csdn.net/missayaaa/article/details/80251823?utm_source=copy 
#版权声明：本文为博主原创文章，转载请附上博文链接！
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io 
import scipy.misc 

#定义一些函数 
#卷积 
def _conv_layer(input, weights, bias): 
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME') 
    return tf.nn.bias_add(conv, bias) 

#池化 
def _pool_layer(input): 
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME') 
    
#减像素均值操作 
def preprocess(image, mean_pixel): 
    return image - mean_pixel 

#加像素均值操作 
def unprocess(image, mean_pixel): 
    return image + mean_pixel 
    
#读 
def imread(path): 
    return scipy.misc.imread(path).astype(np.float) 
    
#保存 
def imsave(path, img): 
    img = np.clip(img, 0, 255).astype(np.uint8) 
    scipy.misc.imsave(path, img) 
print ("Functions for VGG ready") 
#定义VGG的网络结构，用来存储网络的权重和偏置参数 
def net(data_path, input_image):
    #拿到每一层对应的参数 
    layers = ( 
         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 
         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 
         'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 
         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 
         'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 
         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 
         'relu5_3', 'conv5_4', 'relu5_4' 
    ) 
    data = scipy.io.loadmat(data_path) 
    #原网络在训练的过程中，对每张图片三通道都执行了减均值的操作，这里也要减去均值 
    mean = data['normalization'][0][0][0] 
    mean_pixel = np.mean(mean, axis=(0, 1)) 
    #print(mean_pixel) 
    #取到权重参数W和b,这里运气好的话，可以查到VGG模型中每层的参数含义，查不到的 
    #话可以打印出weights，然后打印每一层的shape，推出其中每一层代表的含义 
    weights = data['layers'][0] 
    #print(weights) 
    net = {} 
    current = input_image 
    #取到w和b 
    for i, name in enumerate(layers): 
        #:4的含义是只看每一层的前三个字母，从而进行判断 
        kind = name[:4] 
        if kind == 'conv': 
            kernels, bias = weights[i][0][0][0][0] 
            # matconvnet: weights are [width, height, in_channels, out_channels]\n", 
            # tensorflow: weights are [height, width, in_channels, out_channels]\n", 
            #这里width和height是颠倒的，所以要做一次转置运算 
            kernels = np.transpose(kernels, (1, 0, 2, 3)) 
            #将bias转换为一个维度 
            bias = bias.reshape(-1) 
            current = _conv_layer(current, kernels, bias) 
        elif kind == 'relu': 
            current = tf.nn.relu(current) 
        elif kind == 'pool': 
            current = _pool_layer(current) 
        net[name] = current 
    assert len(net) == len(layers) 
    return net, mean_pixel, layers 
print ("Network for VGG ready") 
#cwd  = os.getcwd() 
#这里用的是绝对路径 
VGG_PATH = "E:/VScode/transstyle/imagenet-vgg-verydeep-19.mat" 
#需要可视化的图片路径，这里是一只小猫 #好了，现在是一艘船
IMG_PATH = "E:/VScode/AirbusShipDetectionVGG19/project_train/train/00f34434e.jpg" 
input_image = imread(IMG_PATH) 
#获取图像shape 
shape = (1,input_image.shape[0],input_image.shape[1],input_image.shape[2]) 
#开始会话 
with tf.Session() as sess: 
    image = tf.placeholder('float', shape=shape) 
    #调用net函数 
    nets, mean_pixel, all_layers = net(VGG_PATH, image) 
    #减均值操作（由于VGG网络图片传入前都做了减均值操作，所以这里也用相同的预处理 
    input_image_pre = np.array([preprocess(input_image, mean_pixel)]) 
    layers = all_layers # For all layers \n", 
    # layers = ('relu2_1', 'relu3_1', 'relu4_1')\n", 
    for i, layer in enumerate(layers): 
        print ("[%d/%d] %s" % (i+1, len(layers), layer)) 
        features = nets[layer].eval(feed_dict={image: input_image_pre}) 
        print (" Type of 'features' is ", type(features)) 
        print (" Shape of 'features' is %s" % (features.shape,)) 
        # Plot response \n", 
        #画出每一层 
        if 1: 
            plt.figure(i+1, figsize=(10, 5)) 
            plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i+1) 
            plt.title("" + layer) 
            plt.colorbar() 
            plt.show()
 
