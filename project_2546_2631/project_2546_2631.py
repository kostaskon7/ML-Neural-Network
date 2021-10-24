#!/usr/bin/env python
# coding: utf-8

# #                                 Project for neuro-fuzzy computing
# 
# 
# ##                         Konstantinos Konstntinidis 2546 Stavrinos Nikolaos 2631
# 
# Για τη παρουσίαση του project μας επιλέξαμε να τρέξουμε τον κώδικα τοπικά στο jupiter notebook σε environment του tensorflow για cpu έτσι ώστε να μπορέσουμε να περιγράψουμε κάθε κομμάτι του κώδικα ξεχωριστά.
# 
# 

# ## Επιλέγουμε να κάνουμε train το μοντέλο μας με χρηση της τιμής open από τη κατανομή

# In[19]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


# Importing the training set
dataset_train = pd.read_excel('Stock_Price_Training_Data.xlsx')
training_set = dataset_train.iloc[:,1:2].values
print(training_set)

train_data = training_set[0:(len(training_set)-30), :]
valid_data = training_set[(len(training_set)-30):, :]


# ## Χρησιμοποιυμε scaling των τιμων απο -1 εως 1 
# καθως με μεγαλες τιμες το training γινεται ασταθες που σημαίνει ότι μπορεί να υποφέρει από κακή απόδοση κατά τη διάρκεια της μάθησης, με αποτέλεσμα υψηλότερο generalization error.

# In[20]:


# Feature Scaling
sc = MinMaxScaler(feature_range = (-1, 1))
training_set_scaled = sc.fit_transform(training_set)


#  ## Για το project χωριζουμε το dataset σε training και testing
#  Το testing αποτελειται απο τις τελευταιες 30 τιμες του dataset.
#  
# # Τα δεδομενα μας θα βασιζονται σε προβλεψη 60 timesteps και 1 output
#  βάση των προηγούμενων 60 ημερών πρόκειται να προβλέψουμε την τιμή της μετοχής στην 61η μέρα και θα γίνει για όλες τις τιμές του training set -60 μέρες
# 

# In[21]:


X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# ## Τα δεδομενα που θα χρησιμοποιηθουν για το validation των 30 τελευταιων τιμων

# In[22]:


model_set = dataset_train.iloc[:, 1:2].values
model_data = model_set[(len(model_set)-len(valid_data)-60):, :]
model_data = model_data.reshape(-1, 1)
model_data_scaled = sc.transform(model_data)

X_test = []
y_test = []
for i in range(60, model_data_scaled.shape[0]):
    X_test.append(model_data_scaled[i-60:i, 0])
    y_test.append(model_data_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)


# ## Reshaping
# τα Frameworks όπως Keras / TensorFlow / PyTorch απαιτούν το σύνολο δεδομένων να γίνει reshaped σε άλλη διάσταση που την καθορίζει η δεύτερη του παράμετρος https://www.tensorflow.org/api_docs/python/tf/reshape

# In[23]:


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# # Building the reccurent neural network for our prediction

# ## Initialising the RNN
# 
# Επιλεξαμε την LSTM καθως μπορει να απομνημονεύσει πληροφορίες για μεγάλα χρονικά διαστήματα, και ειναι χρησιμη για μεγαλα input 
# files με training data,οπως στη δικια μας περιπτωση.
# 
# ## Χρησιμοποιόυμε 4 layers της LSTM
# Το 1ο και 4ο layer εχουν απο 50 νευρωνες,ενω 2ο και 3ο απο 100 με 1 τελικη εξοδο.Επιση το 1ο layer χρησιμοποιει ως συναρτηση ενεργοποιησης την sigmoid.
# Δεν χρησιμοποιησαμε dropout επειδη το LSTM ειναι καλο για long-term υπολογισμους και δεν θελαμε να υπαρχει περιπτωση να χασουμε σημαντικα δεδομενα λογω του dropout.

# In[24]:


regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(
    X_train.shape[1], 1), activation='sigmoid', bias_initializer='glorot_uniform'))

regressor.add(LSTM(units=100,  return_sequences=True,
                   bias_initializer='glorot_uniform'))

regressor.add(LSTM(units=100,  return_sequences=True,
                   bias_initializer='glorot_uniform'))

regressor.add(LSTM(units=50, bias_initializer='glorot_uniform'))

regressor.add(Dense(units=1))


# ## Weight and bias values before training

# In[25]:


for layer in regressor.layers:
  print(layer)
  print("Weights :")
  print(layer.get_weights()[0])
  print("Biases:")
  print(layer.get_weights()[1])


# In[26]:


regressor.summary()


# ## Setting the optimizer
# ## Χρησιμοποιούμε τον [adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) με τις βασικές παραμέτρους του όπως χρησιμοποιόυνται και στο site του tensorflow
# Ο Adam είναι ένας αλγόριθμος βελτιστοποίησης που μπορεί να χρησιμοποιηθεί αντί της διαδικασίας της κλασικής stochastic gradient descent για την ενημέρωση επαναληπτικών βαρών δικτύου με βάση τα δεδομένα εκπαίδευσης.

# In[27]:


optimizer = optimizers.Adam(lr=0.001 , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# ## Loss function
# Ορίζουμε ως loss function μας την [mean_squared_error](https://keras.io/api/losses/regression_losses/#mean_squared_error-function)
# 

# In[28]:


regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')


# ## Start the training
# Επιλεγουμε batch size=32 και epochs=100 μετα απο αρκετες δοκιμες 

# In[29]:


start=time.time()
# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, epochs=100,
                        batch_size=32, validation_data=(X_test, y_test))
end=time.time()


# In[30]:


print("It took '{}' seconds to to train the model".format(np.round(end-start,4)))


# ## Weight and bias values after training

# In[31]:


for layer in regressor.layers:
  print(layer)
  print("Weights :")
  print(layer.get_weights()[0])
  print("Biases:")
  print(layer.get_weights()[1])


# ## Εκτύπωση του loss του μοντελου μας

# In[32]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


# ## Αποτιμηση της επιδοση πανω στα δεδομενα 

# In[33]:


print("\n\n EVALUATION")
start = time.time()
score = regressor.evaluate(X_test, y_test, batch_size=1)
end = time.time()
print("This test uses the validation set and Testing time took '{}' seconds".format(
    np.round(end - start, 4)))

print("\n", regressor.metrics_names,
      "on the validation dataset is " + str(score))


# ## Χρονος αποκρισης υπολογισμου prediction

# In[34]:


print("\n\n Network response time")
start = time.time()
predicted_stock_price = regressor.predict(X_test)
end = time.time()

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

print("This test took '{}' seconds".format(np.round(end - start, 4)))


# ## Ο πραγματικος ελεγχος θα γινει απο τον εξεταστη, για τη παρουσιαση ωστοσο κανουμε validation με τα 30 τελευταια στοιχεια του training set

# In[35]:


plt.plot(valid_data, color='blue', label='Stock Price')
plt.plot(predicted_stock_price, color='green', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# ## Printing the absolute error

# In[36]:


print("\n The mean absolute error is  ", mean_absolute_error(
    valid_data, predicted_stock_price))

