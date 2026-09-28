import tensorflow as tf 
#Define and execute operations on-the-fly 
a=tf.constant(2.0) 
b=tf.constant(3.0) 
c=a+b 
print("Result:", c.numpy())