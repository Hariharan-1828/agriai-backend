"""Download and convert MobileNetV3 Small to TFLite format directly."""
import tensorflow as tf
import urllib.request
import tarfile
import os

print("Downloading SavedModel tarball...")
url = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5?tf-hub-format=compressed"
urllib.request.urlretrieve(url, "mobilenet_v3.tar.gz")

print("Extracting...")
with tarfile.open("mobilenet_v3.tar.gz", "r:gz") as tar:
    tar.extractall("mobilenet_v3_saved_model")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("mobilenet_v3_saved_model")
tflite_model = converter.convert()

with open("mobilenet_v3_small.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Saved mobilenet_v3_small.tflite ({len(tflite_model)} bytes)")
