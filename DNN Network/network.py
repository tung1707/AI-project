import tensorflow as tf
from tensorflow.keras import datasets,layers,models

# 1 network thi can co gi (so luong neuron,so luong layer,activation_function,drop_out)
# activation function
# khoi tao
# so luong layer

#bieu dien qua tensorflow
#bieu dien qua pytorch
new_network = 
tf.add(
    Dense(number_neuron = 15,activation="sigmoid"),
    Dense(number_neuron = 20,activation = "sigmoid"),
    Dense(number_neuron = 10,activation = "relu",drop_out = 0.5)
)

for i in range(10):
    new_network.fit()

#bai chua
model = models.Sequential() #khoi tao model
model.add(layers.Conv2D(32,(3,3),activation='sigmoid',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2d(32,(3,3),activation="sigmoid"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2d(32,(3,3),activation='sigmoid'))
model.add(layers.MaxPooling2D(1,1))
model.add(layers.Conv2d(32,(3,3),activation='sigmoid'))
model.summary()


