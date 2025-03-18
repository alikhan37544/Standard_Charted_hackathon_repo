import mtcnn
import inspect

print("MTCNN Version:", mtcnn.__version__ if hasattr(mtcnn, "__version__") else "Unknown")
print("\nMTCNN Constructor Signature:")
print(inspect.signature(mtcnn.MTCNN.__init__))
print("\nAccepted parameters:")
print(inspect.getfullargspec(mtcnn.MTCNN.__init__))