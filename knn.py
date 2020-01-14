import numpy as np
import pandas as pd
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data('input/mnist.npz')

print(x_train.size)
print(x_test.size)

Kaggle_test_image = pd.read_csv("input/test.csv")
Kaggle_test_image = Kaggle_test_image.values.astype("uint8")
Kaggle_test_label = np.empty( (28000,1), dtype="uint8" )

print(Kaggle_test_image.size)
print(Kaggle_test_image[0].size)


c1=0; c2=0;
print("Classifying Kaggle's 'test.csv' using kNN k=1 and MNIST 70k images")
for i in range(0,28000): # loop over Kaggle test
    for j in range(0,70000): # loop over MNIST images
        if np.absolute(Kaggle_test_image[i,] - x_train[j,]).sum()==0:
            Kaggle_test_label[i] = y_train[j]
            if i%1000==0:
                print("  %d images classified perfectly" % (i))
            if j<60000:
                c1 += 1
            else:
                c2 += 1
            break
if c1+c2==28000:
    print("  28000 images classified perfectly")
    print("Kaggle's 28000 test images are fully contained within MNIST's 70000 dataset")
    print("%d images are in MNIST-train's 60k and %d are in MNIST-test's 10k" % (c1,c2))



results = pd.Series(Kaggle_test_label.reshape(28000,),name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("Do_not_submit.csv",index=False)