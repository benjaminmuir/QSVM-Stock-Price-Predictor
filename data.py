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

def calculate_atr(data, window=14):
    """calculate Average True Range"""
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range.rolling(window).mean()


def calculate_rsi(prices, window=100):
    """Small helper function for getting RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def analyze_stock_characteristics(symbol):
    """
    done for testing purposes, displays metrics of the stocks and how they react 
    """
    # Load data
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'stocks', f'{symbol}.csv')
    data = pd.read_csv(csv_path, parse_dates=['Date'])
    
    # Calculate basic statistics
    daily_returns = data['Close'].pct_change().dropna()
    
    # Volatility metrics
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Trend strength - using autocorrelation
    autocorr = daily_returns.autocorr(lag=1)
    
    # Volume-price correlation
    volume_price_corr = data['Close'].corr(data['Volume'])
    
    # Calculate technical indicator effectiveness
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    
    # Measure how well MACD predicts direction
    data['Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['MACD_Signal'] = (data['MACD'] > data['MACD'].shift(1)).astype(int)
    macd_accuracy = (data['MACD_Signal'] == data['Direction']).mean()
    
    # Volume's predictive power
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Volume_Signal'] = (data['Volume_Change'] > 0).astype(int)
    volume_accuracy = (data['Volume_Signal'] == data['Direction']).mean()
    
    return {
        'Symbol': symbol,
        'Volatility': volatility,
        'Autocorrelation': autocorr,
        'Volume-Price Correlation': volume_price_corr,
        'MACD Accuracy': macd_accuracy,
        'Volume Accuracy': volume_accuracy
    }

# Analyze all stocks
results = []
for symbol in ['A', 'MSFT', 'AAPL', 'NVDA']:
    try:
        stats = analyze_stock_characteristics(symbol)
        if 'Symbol' not in stats:
            stats['Symbol'] = symbol
        results.append(stats)
        print(f"Analyzed {symbol}")
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")

# Create comparison table
comparison_df = pd.DataFrame(results)

# Verify columns
print("DataFrame columns:", comparison_df.columns.tolist())
print(comparison_df)

# Plot comparison - make sure to use existing columns
plt.figure(figsize=(12, 8))

# Get columns excluding 'Symbol'
cols_to_plot = [col for col in comparison_df.columns if col != 'Symbol']

for i, col in enumerate(cols_to_plot):
    plt.subplot(2, 3, i+1)
    plt.bar(comparison_df['Symbol'], comparison_df[col])
    plt.title(col)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('stock_comparison.png')
plt.show()


#Relative file path to the CSV file
stock_symbols = ['A']
for symbol in stock_symbols:
    print(f"\nTesting with stock ticker: {symbol}")
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'stocks', f'{symbol}.csv')
    db = pd.read_csv(csv_path, parse_dates=['Date'])

    
    db['Prev_Close'] = db['Close'].shift(1)

    #stock market indicators from the paper 
    db['EMA_12'] = db['Close'].ewm(span=12, adjust=False).mean()  
    db['EMA_26'] = db['Close'].ewm(span=26, adjust=False).mean()  
    db['MACD'] = db['EMA_12'] - db['EMA_26'] 
    db['ATR'] = calculate_atr(db, 14) 
    db['Price_Volume_Ratio'] = db['Close'] / db['Volume']
    #using RSI as feature, not sure if needed just yet
    db['RSI'] = calculate_rsi(db['Close'],14)
    db['EMA_26/EMA_12'] = db['EMA_26']/db['EMA_12']
    db['Momentum'] = db['Close'] / db['Prev_Close']
    #remove first row (doesnt have prev values)
    db = db.dropna()


    #samples used for model. 100-200 will run in under 1 minute 
    #Anything @ 500 or more can take upwards of 5-10 minutes
    max_samples = 250
    db_sample = db.iloc[-max_samples:]
    sample_size = (len(db_sample))

    #using minmaxscaler to normalize the data (same as paper)
    scaler_features = MinMaxScaler()

    #feature combinations for testing 
    """
    Optimal Feature combinations seem to be: 
    Momentum, Volume >>>>>>>>>
    Prev_Close, Volume 
    MACD, Volume 

    F-score and accuracy seem to depend pretty heavily on the stock ticker
    All combinations perform very well (high 60-70s) on stock A.csv
    Certain features perform well on the rest but are not nearly as high as A.csv. 
    """
    feature_combinations = [
        #New combinations with EMAs and raw features
        ['Prev_Close', 'Volume'], #works well with A and NVDA 
        ['MACD', 'Volume'],  #same as prev_close
        ['EMA_12', 'EMA_26', 'Volume'], 
        ['RSI', 'Volume'], #works well with MSFT
        ['Momentum', 'Volume'], #works well with everything but MSFT (low f-score, high acc)
        ['MACD', 'EMA_26/EMA_12'],  
        ['Price_Volume_Ratio', 'MACD']  
    ]
    
    

    for features in feature_combinations:
        #Using 80/20 train test split with chronological splitting maybe update for validation testing as well? 
        train_size = int(0.8 * sample_size)

        #binary classification (1 if price went up, 0 if decrased)
        y_close_direction = (db_sample['Close'] > db_sample['Prev_Close']).astype(int)

        y_train_direction, y_test_direction = y_close_direction[:train_size], y_close_direction[train_size:]
        
        #apply scaler to raw feature data
        fs = scaler_features.fit_transform(db_sample[features])

        print(f"Testing feature combination: {features}")
        
        X_train, X_test = fs[:train_size], fs[train_size:]
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

        quantum_kernel = FidelityQuantumKernel(
            feature_map=feature_map, 
            fidelity=fidelity
        )


        #qsvm model training 
        qsvc = QSVC(quantum_kernel=quantum_kernel, C=5.0)
        qsvc.fit(X_train,y_train_direction)

        #predicting 
        direction_pred = qsvc.predict(X_test)
        #get accuracy value and f1_score (for comparison with paper)
        accuracy = qsvc.score(X_test, y_test_direction)
        f1 = f1_score(y_test_direction, direction_pred, average='binary')

        print(f"F1 Score: {f1:.2f}")
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

#Get previous day close prices for test set
prev_close_test = db_sample['Prev_Close'].iloc[train_size:].values
close_pred = direction_to_price(prev_close_test, direction_pred)

#Plot results
plt.figure(figsize=(12, 8))

#Get actual dates and prices for visualization
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


