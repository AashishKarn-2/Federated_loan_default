import os
import pandas as pd
import flwr as fl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()




data = pd.read_csv("Fin.csv")
data.drop(["Index"] , inplace =  True , axis = 1)
X = data.drop("Defaulted?" , axis = 1)
Y = data["Defaulted?"]
X_train , X_test , y_train , y_test = train_test_split(X , Y , test_size = 0.3)


#scale x_train and x_test

scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#define model

model = Sequential()

model.add(Dense(11,activation='relu',input_dim=3))
model.add(Dense(11,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])




# Define Flower client
class BankDefault(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train_scale, y_train, epochs=1, batch_size=32,verbose=1,validation_split=0.2)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test_scale, y_test)
        return loss, len(X_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=BankDefault())