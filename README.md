# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.datasets import sunspots

data = sunspots.load_pandas().data
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')
data.set_index('YEAR', inplace=True)

plt.plot(data.index, data['SUNACTIVITY'])
plt.xlabel('Year')
plt.ylabel('Sunspot Activity')
plt.title('Sunspot Activity Time Series')
plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['SUNACTIVITY'])

plot_acf(data['SUNACTIVITY'])
plt.show()

plot_pacf(data['SUNACTIVITY'])
plt.show()

train_size = int(len(data) * 0.8)
train, test = data['SUNACTIVITY'][:train_size], data['SUNACTIVITY'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 11))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.xlabel('Year')
plt.ylabel('Sunspot Activity')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()

```
### OUTPUT:
<img width="619" height="460" alt="image" src="https://github.com/user-attachments/assets/b4b1f504-c94e-480e-ae5a-f0c66145b938" />

<img width="593" height="430" alt="image" src="https://github.com/user-attachments/assets/30032031-7dc8-4e1b-818f-2f77c930d908" />

<img width="603" height="426" alt="image" src="https://github.com/user-attachments/assets/6ebef73e-42c4-49cc-9f6b-e713164431dc" />

<img width="618" height="485" alt="image" src="https://github.com/user-attachments/assets/12dbf078-8065-42b5-a0ea-3c61f6881ad4" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
