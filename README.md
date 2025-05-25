# QSVM-Stock-Price-Predictor

This project was created to attempt to improve upon the ideas of certain research papers studying the optimization of machine learning and its effect on stock portfolios. 

The program can perform prediction on a number of stocks, those include: AAPL, JNJ, V, and HON. If you wish to add additional stocks for testing purposes, the yfinance library can be used to download statistic for any NASDAQ-traded stock. 

The program functions by first taking in data in the form of a pandas dataframe. This data is normalized and a number of common stock indicators are calculated for use later in the code. This includes MACD, EMA, ATR, and RSI.  

## SVM Algorithm
The algorithm levrages what's known as a Support Vector Machine, which is a machine learning algorithm designed to aid in creating linearly seperable data from data which can't be seperated linearly. Once the data is linearly seperable, it becomes much easier to create a binary classification for said data. In our case, we predict the classifying data as the stock price going 'up' or going 'down'. 

Our SVM utilizes an 80/20 train/test split to ensure that we are allowing the machine to handle as much data as possible, while still maintaining good coverage of testing data. 

## Becoming a Quantum Support Vector Machine
Creating a Quantum Support Vector Machine allows the predictions to handle much higher amount of data, as we are able to handle many different computations simultaneously. 

After generating our normalized feature data, the algorithm will use a ZZ Feature Map to enhance the data. This ZZ feature map will apply a Quantum Circuit that perform specific operations on each of the feature variables.
![zzfeature](https://github.com/user-attachments/assets/c9a4784e-1c56-47b4-ba6d-29a63ad23e59)
As you can see, the x[0,1,2] are the features (Stock Indicators). A series of operations, Hadamard Gates, and X Gates are applied to these features. 
The output is a map of this feature data in a higher dimensional space. 

## Quantum Kernel 
Now that we have Quantum Feature Data, the algorithm seperates the data linearly using a quantum kernel. ![image](https://github.com/user-attachments/assets/aae6a45c-68fd-4622-9495-bde77d74a2e7)
This Quantum Kernel calculates the ideal line to "slice" our data in half, allowing for binary classification to occur. 

## QSVM Prediction 
Once the data is linearly seperated in the higher dimension space, the machine can now predict the stock price using the data. We used Qiskit's QSVC function to 
process the stock price predictions. Accuracy and f1_score are the two metrics used to compare with research papers. 

## Additional Features
There is a data_testing file that includes most of the same code as the original. However it introduces the ability to test the feature combinations in bulk, allowing for manual feature selection. 
