from skimage import io,transform 
import tensorflow as tf 
import numpy as np 
path1 = "./test/daisy.jpg" 
path2 = "./test/sunflower.jpg" 

flower_dict = {0:'dasiy',1:'sunflowers'} 

w=100 
h=100 
c=3 

def read_one_image(path): 
    img = io.imread(path) 
    img = transform.resize(img,(w,h)) 
    return np.asarray(img) 

with tf.Session() as sess: 
    data = [] 
    data1 = read_one_image(path1) 
    data2 = read_one_image(path2) 
    data.append(data1) 
    data.append(data2) 

    
    saver = tf.train.import_meta_graph('./train/model.ckpt.meta') 
    saver.restore(sess,tf.train.latest_checkpoint('./train/')) 
    
    graph = tf.get_default_graph() 
    x = graph.get_tensor_by_name("x:0") 
    feed_dict = {x:data} 
    
    logits = graph.get_tensor_by_name("logits_eval:0") 
    classification_result = sess.run(logits,feed_dict) 
    
    #plot result 
    print(classification_result) 
    print(tf.argmax(classification_result,1).eval()) 
    output = [] 
    output = tf.argmax(classification_result,1).eval() 
    for i in range(len(output)): 
        print("the",i+1,"flower is predicted to be:"+flower_dict[output[i]]) 
