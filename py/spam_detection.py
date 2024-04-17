# Import Package
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,BatchNormalization,LSTM,Embedding
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read Dataset
df = pd.read_csv("../dataset/spam.csv",encoding='ISO-8859-1')
df = df.drop(["Unnamed: 2",'Unnamed: 3',"Unnamed: 4"],axis = 1)
print(df.head())

# Variable value
x = df["v2"]
y = df["v1"]

# Encoder Label
lebel_encoder = LabelEncoder()
y = lebel_encoder.fit_transform(y)

# Split Train and Test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.1)

# Tokenizer
max_words = 500
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
seq = tokenizer.texts_to_sequences(x_train)
seq_2 = tokenizer.texts_to_sequences(x_test)

# Padding Sequences
max_len = 100
padded_docs = pad_sequences(seq,padding='pre',maxlen = max_len)
padded_docs2 = pad_sequences(seq_2,padding='pre',maxlen = max_len)

# Model
model = keras.Sequential()
model.add(Embedding(max_words,30,input_length=max_len))
model.add(LSTM(256))
model.add(Dense(124,activation="relu"))
model.add(keras.layers.Dropout(.4))
model.add(Dense(124,activation="relu"))
model.add(keras.layers.Dropout(.4))
model.add(Dense(124,activation="relu"))
model.add(keras.layers.Dropout(.4))
model.add(Dense(1,activation="sigmoid"))

# Compile Model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

# Train Model
hist = model.fit(padded_docs,y_train,epochs=50,batch_size=512,validation_data=(padded_docs2,y_test), validation_split=0.2)

# Save Model
model.save("spam-detection.h5")

# Accuracy
model.evaluate(padded_docs2,y_test)
acc = hist.history['accuracy']
acc_val = hist.history['val_accuracy']
loss = hist.history['loss']
loss_val = hist.history['val_loss']

# Plot
plt.plot(acc,color="red",label="accuracy")
plt.plot(acc_val,color="green",label="validation accuracy")
plt.plot(loss,color="blue",label="loss")
plt.plot(loss_val,color="orange",label="validation loss")
plt.legend()
plt.show()

# History
plt.plot(hist.history['val_loss'])
plt.show()