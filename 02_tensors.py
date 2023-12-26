import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

print(tf.__version__)

# x = tf.constant(4, shape = (1,1), dtype = tf.float32)
# print(x)

# x = tf.constant([[1,2,3], [3,4,5]])
# print(x)

# x = tf.ones(3,3)
# print(x)

# x = tf.eye(3)
# print(x)

x = tf.random.normal((3,3), mean = 0, stddev = 1)
print(x)