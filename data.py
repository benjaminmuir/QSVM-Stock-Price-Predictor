from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from qiskit.circuit.library import ZZFeatureMap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

def direction_to_price(prev_prices, directions):
    """Convert directional predictions to price predictions
    based on binary outcomes from model
    """
    predictions = []
    for i, direction in enumerate(directions):
        prev_price = prev_prices[i]
        avg_change = np.abs(np.diff(db_sample['Close'][:train_size])).mean()
        
        if direction == 1:  # Up
            predictions.append(prev_price + avg_change)
        else:  # Down
            predictions.append(prev_price - avg_change)
    return predictions

def calculate_atr(data, window=14):
    """calculate Average True Range"""
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range.rolling(window).mean()


def calculate_rsi(prices, window=14):
    """Small helper function for getting RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

stock_symbols = ['A', 'AAPL', 'HON', 'JNJ', 'MSFT', 'NVDA', 'V']

#Relative file path to the CSV file
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, 'stocks', f'{'A'}.csv')
db = pd.read_csv(csv_path, parse_dates=['Date'])

#need to use previous values to avoid data leakage
db['Prev_Close'] = db['Close'].shift(1)
db['Prev_Volume'] = db['Volume'].shift(1)
db['Momentum'] = db['Close'] / db['Prev_Close']
db['Momentum_Prev'] = db['Momentum'].shift(1)

#stock market indicators from the paper 
db['EMA_12'] = db['Close'].ewm(span=12, adjust=False).mean()  
db['EMA_26'] = db['Close'].ewm(span=26, adjust=False).mean()  
db['MACD'] = db['EMA_12'] - db['EMA_26'] 
db['ATR'] = calculate_atr(db, 14) 
db['Price_Volume_Ratio'] = db['Close'] / db['Volume']
db['RSI'] = calculate_rsi(db['Close'],14)
db['EMA_26/EMA_12'] = db['EMA_26']/db['EMA_12']

#remove first row (doesnt have prev values)
db = db.dropna()

#samples used for model
max_samples = 200
db_sample = db.iloc[-max_samples:]
sample_size = (len(db_sample))

#using minmaxscaler to normalize the data (same as paper)
scaler = MinMaxScaler()

feature_combinations = [
    #simple measure
    ['Prev_Close', 'Volume_Prev'],
    #rsi and vol prev perform high, add atr to make better?
    ['RSI', 'Volume_Prev', 'ATR'],
    #major stock indicators and vol prev
    ['EMA_26/EMA_12', 'Volume_Prev'],
    ['RSI', 'Volume_Prev'],
    ['ATR', 'Volume_Prev'],
    ['MACD', 'Volume_Prev'],
    #all stock indicators together
    ['MACD', 'ATR', 'RSI', 'EMA_26/EMA_12'],
    ['MACD', 'ATR', 'RSI','Volume_Prev'],
    ['MACD', 'ATR', 'RSI'],
    ['MACD', 'EMA_26/EMA_12'],
    ['Price_Volume_Ratio', 'MACD']
]

#Using 80/20 train test split
train_size = int(0.8 * sample_size)

test_dates = db_sample['Date'].iloc[train_size:].values
actual_close = db_sample['Close'].iloc[train_size:].values
prev_close_test = db_sample['Prev_Close'].iloc[train_size:].values

#binary classification (1 if price went up, 0 if decrased)
y_close_direction = (db_sample['Close'] > db_sample['Prev_Close']).astype(int)
y_train_direction, y_test_direction = y_close_direction[:train_size], y_close_direction[train_size:]

accuracy_results = []
feature_labels = []

for features in feature_combinations:
    #fit scaler onto train data ONLY, not onto test data (don't want data leakage to model)
    X_train = scaler.fit_transform(db_sample[features][:train_size])
    X_test = scaler.transform(db_sample[features][train_size:])

    print(f"Testing feature combination: {features}")
    #determine x values for classification
    print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")

    #initalize a feature map (entanglement : linear best? simpler circuit = better performance)
    feature_map = ZZFeatureMap(
        feature_dimension=len(features),
        reps=2, entanglement='linear'
    )

    print("\nFeature map circuit template:")
    print(feature_map.decompose().draw())

    #initalize quantum kernel 
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)

    #compute quantum kernel to linearly seperate feature map
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map, 
        fidelity=fidelity
    )

    #qsvm model training using quantum kernel
    qsvc = QSVC(quantum_kernel=quantum_kernel, C=5.0)
    #apply qsvc onto data
    qsvc.fit(X_train,y_train_direction)

    #predicting with 20% sample size test data
    direction_pred = qsvc.predict(X_test)
    #get accuracy value and f1_score (for comparison with paper)
    accuracy = qsvc.score(X_test, y_test_direction)
    f1 = f1_score(y_test_direction, direction_pred, average='binary')

    print(f"F1 Score: {f1:.3f}")
    print(f"\nAccuracy: {accuracy:.3f}")

    # Store results for the bar graph
    accuracy_results.append(accuracy)
    feature_labels.append(', '.join(features))  # Create readable labels

    #Get previous day close prices for test set
    prev_close_test = db_sample['Prev_Close'].iloc[train_size:].values
    close_pred = direction_to_price(prev_close_test, direction_pred)


