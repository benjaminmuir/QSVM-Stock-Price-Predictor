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

"""
This is used for testing right now, will be deleted in final version 

"""



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

# Fixed hyperparameters
SAMPLE_SIZE = 200
C_VALUE = 5.0
REPS = 2
ENTANGLEMENT = 'linear'

# Stock tickers to test
stock_symbols = ['AAPL', 'HON', 'JNJ','V']

# Feature combinations to test
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

# Results structure
results = {
    'Stock': [],
    'Features': [],
    'Accuracy': [],
    'F1_Score': []
}

# Process each stock
for stock in stock_symbols:
    print(f"\n===== Testing stock: {stock} =====")
    
    # Load and prepare data
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'stocks', f'{stock}.csv')
    
    try:
        db = pd.read_csv(csv_path, parse_dates=['Date'])
        db['Volume_Prev'] = db['Volume'].shift(1)
        # Calculate indicators
        db['Prev_Close'] = db['Close'].shift(1)
        db['EMA_12'] = db['Close'].ewm(span=12, adjust=False).mean()  
        db['EMA_26'] = db['Close'].ewm(span=26, adjust=False).mean()  
        db['MACD'] = db['EMA_12'] - db['EMA_26'] 
        db['ATR'] = calculate_atr(db, 14) 
        db['Price_Volume_Ratio'] = db['Close'] / db['Volume']
        db['RSI'] = calculate_rsi(db['Close'], 14)
        db['EMA_26/EMA_12'] = db['EMA_26']/db['EMA_12']
        db['Momentum'] = db['Close'] / db['Prev_Close']
        db['Momentum_Prev'] = db['Momentum'].shift(1)
        
        # Remove rows with NaN values
        db = db.dropna()
        
        # Get the last SAMPLE_SIZE samples
        db_sample = db.iloc[-SAMPLE_SIZE:]
        sample_size = len(db_sample)
        train_size = int(0.8 * sample_size)
        
        # Binary classification target
        y_close_direction = (db_sample['Close'] > db_sample['Prev_Close']).astype(int)
        y_train_direction, y_test_direction = y_close_direction[:train_size], y_close_direction[train_size:]
        
        # Test each feature combination
        for features in feature_combinations:
            feature_name = ', '.join(features)
            print(f"\nTesting feature combination: {feature_name}")
            
            # Scale features
            try:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(db_sample[features][:train_size])
                X_test = scaler.transform(db_sample[features][train_size:])
                
                # Create feature map
                feature_map = ZZFeatureMap(
                    feature_dimension=len(features),
                    reps=REPS,
                    entanglement=ENTANGLEMENT
                )
                
                # Initialize quantum kernel
                sampler = StatevectorSampler()
                fidelity = ComputeUncompute(sampler=sampler)
                quantum_kernel = FidelityQuantumKernel(
                    feature_map=feature_map,
                    fidelity=fidelity
                )
                
                # Train and evaluate model
                qsvc = QSVC(quantum_kernel=quantum_kernel, C=C_VALUE)
                qsvc.fit(X_train, y_train_direction)
                
                # Get predictions and metrics
                direction_pred = qsvc.predict(X_test)
                accuracy = qsvc.score(X_test, y_test_direction)
                f1 = f1_score(y_test_direction, direction_pred, average='binary')
                
                print(f"Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}")
                
                # Store results
                results['Stock'].append(stock)
                results['Features'].append(feature_name)
                results['Accuracy'].append(accuracy)
                results['F1_Score'].append(f1)
                
            except Exception as e:
                print(f"Error with features {feature_name}: {str(e)}")
                
    except Exception as e:
        print(f"Error processing stock {stock}: {str(e)}")

# Convert results to DataFrame for easier visualization
results_df = pd.DataFrame(results)

# Create a pivot table for heatmap visualization
accuracy_pivot = results_df.pivot(index='Features', columns='Stock', values='Accuracy')
f1_pivot = results_df.pivot(index='Features', columns='Stock', values='F1_Score')

# Create visualizations
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.title('Accuracy by Feature Combination and Stock', fontsize=16)
heatmap = plt.imshow(accuracy_pivot, cmap='YlGnBu', aspect='auto')
plt.colorbar(heatmap, label='Accuracy')
plt.xticks(np.arange(len(accuracy_pivot.columns)), accuracy_pivot.columns, rotation=45)
plt.yticks(np.arange(len(accuracy_pivot.index)), accuracy_pivot.index)

# Add text annotations
for i in range(len(accuracy_pivot.index)):
    for j in range(len(accuracy_pivot.columns)):
        value = accuracy_pivot.iloc[i, j]
        if not np.isnan(value):
            plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                    color='black' if value < 0.7 else 'white')

plt.subplot(2, 1, 2)
plt.title('F1 Score by Feature Combination and Stock', fontsize=16)
heatmap = plt.imshow(f1_pivot, cmap='YlGnBu', aspect='auto')
plt.colorbar(heatmap, label='F1 Score')
plt.xticks(np.arange(len(f1_pivot.columns)), f1_pivot.columns, rotation=45)
plt.yticks(np.arange(len(f1_pivot.index)), f1_pivot.index)

# Add text annotations
for i in range(len(f1_pivot.index)):
    for j in range(len(f1_pivot.columns)):
        value = f1_pivot.iloc[i, j]
        if not np.isnan(value):
            plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                    color='black' if value < 0.7 else 'white')

plt.tight_layout()
plt.savefig('feature_stock_comparison.png')
plt.show()

# Also create a bar chart showing average performance across stocks
avg_by_feature = results_df.groupby('Features')[['Accuracy', 'F1_Score']].mean().reset_index()
avg_by_feature = avg_by_feature.sort_values('Accuracy', ascending=False)

plt.figure(figsize=(14, 8))
bars = plt.bar(avg_by_feature['Features'], avg_by_feature['Accuracy'], color='steelblue')
plt.xlabel('Feature Combination', fontsize=12)
plt.ylabel('Average Accuracy Across Stocks', fontsize=12)
plt.title('Average QSVM Accuracy by Feature Combination', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', alpha=0.3)

# Add accuracy values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{avg_by_feature["Accuracy"].iloc[i]:.3f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('average_feature_performance.png')
plt.show()

# Print summary of best performers
print("\n===== Feature Performance Summary =====")
for stock in stock_symbols:
    stock_data = results_df[results_df['Stock'] == stock]
    best_feature = stock_data.loc[stock_data['Accuracy'].idxmax()]
    print(f"\n{stock} - Best performing feature: {best_feature['Features']}")
    print(f"Accuracy: {best_feature['Accuracy']:.3f}, F1 Score: {best_feature['F1_Score']:.3f}")

print("\n===== Overall Top Feature Combinations =====")
top_features = avg_by_feature.head(3)
for i, row in top_features.iterrows():
    print(f"{row['Features']} - Avg Accuracy: {row['Accuracy']:.3f}, Avg F1: {row['F1_Score']:.3f}")