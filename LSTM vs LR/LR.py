import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, mean_squared_error

# Read data, change data from string to float
data = pd.read_csv('NFLX.csv')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

#Change Close data into data that fits model (0 or 1)
data['PriceChange'] = data['Close'] - data['Open']
data['Label'] = (data['PriceChange'] > 0).astype(int)
X = data.drop(['Label', 'Date', 'PriceChange'], axis=1)
y = data['Label']

#Split the data into training and testing sets, train data
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(xTrain, yTrain)

#Logistic Regression accuracy
Pred = model.predict(xTest)
probability = model.predict_proba(xTest)[:, 1]
yProb = np.round(probability)
rmse = np.sqrt(mean_squared_error(yTest, probability))
print(f'RMSE:', rmse)
print('Accuracy:', model.score(xTest,yTest))

#plot
precision, recall, _ = precision_recall_curve(yTest, probability)
avg_precision = average_precision_score(yTest, Pred)
plt.figure(figsize=(5, 4))
plt.plot(recall, precision, label=f'Avg Precision = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()