import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)

# Create a simple tensor
x = tf.constant([[1., 2.], [3., 4.]])
print("\nInput tensor:")
print(x)

# Perform a matrix multiplication
y = tf.matmul(x, x)
print("\nMatrix multiplication result:")
print(y)

# Test GPU availability
print("\nGPU devices:", tf.config.list_physical_devices('GPU'))