import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('MSFT.csv')
closingPrices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
closingPrices = scaler.fit_transform(closingPrices)

#Split data into training and testing
trainSize = int(len(closingPrices) * 0.8)
trainData = closingPrices[:trainSize]
testData = closingPrices[trainSize:]

#Training and testing
def create_sequences(data, seqLength):
    sequences = []
    labels = []
    for i in range(len(data) - seqLength):
        seq = data[i:i+seqLength]
        label = data[i+seqLength]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)
seqLength = 10
xTrain, yTrain = create_sequences(trainData, seqLength)
xTest, yTest = create_sequences(testData, seqLength)

#Build and train LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seqLength, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(xTrain, yTrain, epochs=50, batch_size=32)

#Evaluate model
pred = model.predict(xTest)
predictions = scaler.inverse_transform(pred)
yTest = scaler.inverse_transform(yTest)
rmse = np.sqrt(mean_squared_error(yTest, pred))
print("RMSE:", rmse)

#plot
plt.figure(figsize=(8, 5))
plt.plot(predictions, label='Predicted')
plt.plot(yTest, label='Actual')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.show()

