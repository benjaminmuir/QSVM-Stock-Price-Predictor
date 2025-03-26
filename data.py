#For MinMax normalization of classical data
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def calculate_rsi(prices, window=14):
    """Small helper function for getting RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


db = pd.read_csv('C:/Users/bmuir/QSVM-Stock-Price-Predictor/stocks/A.csv', parse_dates=['Date'])

#create features for previous close and volume
db['Prev_Close'] = db['Close'].shift(1)
db['Prev_Volume'] = db['Volume'].shift(1)
#using RSI as feature, not sure if needed just yet
db['RSI'] = calculate_rsi(db['Close'],14)
#remove first row (doesnt have prev values)
db = db.dropna()


#samples used for model. 100-200 will run in under 1 minute 
#Anything @ 500 or more can take upwards of 5-10 minutes
max_samples = 100
db_sample = db.iloc[-max_samples:]
sample_size = (len(db_sample))

#using minmaxscaler to normalize the data (same as paper)
scaler_features = MinMaxScaler()

feature_columns = ['Prev_Close', 'Prev_Volume','RSI']

#apply scaler to raw feature data
features = scaler_features.fit_transform(db_sample[feature_columns])

#binary classification (1 if price went up, 0 if decrased)
y_close_direction = (db_sample['Close'] > db_sample['Prev_Close']).astype(int)

#Using 80/20 train test split with chronological splitting 
train_size = int(0.8 * sample_size)
X_train, X_test = features[:train_size], features[train_size:]
y_train_direction, y_test_direction = y_close_direction[:train_size], y_close_direction[train_size:]

print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")


#initalize a feature map with 2 features
feature_map = ZZFeatureMap(feature_dimension=len(feature_columns),reps=2)
print("\nFeature map circuit template:")
print(feature_map.decompose().draw())

#initalize quantum kernel 
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

#qsvm model training 
qsvc = QSVC(quantum_kernel=quantum_kernel, C=10.0)
qsvc.fit(X_train,y_train_direction)

#predicting 
direction_pred = qsvc.predict(X_test)
accuracy = qsvc.score(X_test, y_test_direction)

print(f"\nAccuracy: {accuracy:.2f}")




# Convert directional predictions to price predictions
def direction_to_price(prev_prices, directions):
    """Convert directional predictions to price predictions"""
    predictions = []
    for i, direction in enumerate(directions):
        prev_price = prev_prices[i]
        avg_change = np.abs(np.diff(db_sample['Close'][:train_size])).mean()
        
        if direction == 1:  # Up
            predictions.append(prev_price + avg_change)
        else:  # Down
            predictions.append(prev_price - avg_change)
    return predictions

# Get previous day close prices for test set
prev_close_test = db_sample['Prev_Close'].iloc[train_size:].values
close_pred = direction_to_price(prev_close_test, direction_pred)

# Plot results
plt.figure(figsize=(12, 8))

# Get actual dates and prices for visualization
test_dates = db_sample['Date'].iloc[train_size:].values
actual_close = db_sample['Close'].iloc[train_size:].values

plt.plot(test_dates, actual_close, 'b-', label='Actual Close Price')
plt.plot(test_dates, close_pred, 'r--', label='QSVM Predicted Close Price')
plt.title('Stock Close Price: Actual vs. Quantum SVM Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

