import tensorflow as tf

print("TensorFlow version:")
print(tf.__version__)

print("\nAvailable GPU devices:")
print(tf.config.list_physical_devices("GPU"))

print("\nTensorFlow is working correctly!")