## Maximizing Revenue for Taxi Cab Drivers through Payment Type Analysis

### Problem Statement

In the fast paced taxi booking sector, making the most of revenue is essential for long term success and driver happiness. Our goal is to use data-driven
insights to maximize revenue streams for taxi drivers in order to meet this need. Our research aims to determine whether payment methods have an impact on
fare pricing by focusing on the relationship between payment type and fare amount.

### Objective

This project's main goal is to run an A/B test to examine the relationship between the total fare and the method of payment. We use Python hypothesis 
testing and descriptive statistics to extract useful information that can help taxi drivers generate more cash. In particular, we want to find out if there is a big difference in the fares for those who pay with credit cards versus those who pay with cash.

### Research Question

Is there a relationship between total fare amount and payment type and can we nudge customers towards payment methods that generate higher revenue for
drivers, without negatively impacting customer experience?

### Importing Libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')
```

### Loading the dataset


```python
df = pd.read_csv("yellow_tripdata_2020-01.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>congestion_surcharge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2020-01-01 00:28:15</td>
      <td>2020-01-01 00:33:03</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>N</td>
      <td>238</td>
      <td>239</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.5</td>
      <td>1.47</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>11.27</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2020-01-01 00:35:39</td>
      <td>2020-01-01 00:43:04</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>N</td>
      <td>239</td>
      <td>238</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>0.5</td>
      <td>1.50</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>12.30</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2020-01-01 00:47:41</td>
      <td>2020-01-01 00:53:52</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>N</td>
      <td>238</td>
      <td>238</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>10.80</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2020-01-01 00:55:23</td>
      <td>2020-01-01 01:00:14</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>N</td>
      <td>238</td>
      <td>151</td>
      <td>1.0</td>
      <td>5.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.36</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>8.16</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>2020-01-01 00:01:58</td>
      <td>2020-01-01 00:04:16</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>N</td>
      <td>193</td>
      <td>193</td>
      <td>2.0</td>
      <td>3.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.80</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# Exploratory Data Analysis

#### ***VendorID*** -- A code indicating the TPEP provider that provided the record.
    
#### ***tpep_pickup_datetime*** -- The date and time when the meter was engaged.

#### ***tpep_dropoff_datetime*** -- The date and time when the meter was disengaged.

#### ***passenger_count*** -- The number of passengers in the vehicle. This is a driver-entered value.

#### ***trip_distance*** -- The elapsed trip distance in miles reported by the taximeter.

#### ***PULOCATIONID*** -- TLC taxi zone in which the taximeter was engaged.

#### ***DOLOCATIONID*** -- TLC taxi zone in which the taximeter was engaged.

#### ***RateCodeID*** -- The final rate code in effect at the end of the trip.

#### ***store_and_fwd_flag*** -- This flag indicates whether the trip record was held in vehicle memory before sending to the vendor, aka “store

####                           and forward,” because the vehicle did not have a connection to the server.

#### ***payment_type*** -- A numeric code signifying how the passenger paid for the trip.

#### ***fare_amount*** -- The time-and-distance fare calculated by the meter.

#### ***extra*** -- Miscellaneous extras and surcharges. Currently, this only includes. The $0.50  and  $1 rush hour and overnight charges.

#### ***mta_tax*** -- $0.50 MTA tax that is automatically triggered based on the metered rate in use.

#### ***tip_amount*** -- Tip amount – This field is automatically populated for credit card tips.Cash tips are not included.

#### ***tolls_amount*** -- Total amount of all tolls paid in trip.

#### ***improvement_surcharge*** -- $0.30 improvement surcharge assessed trips at the flag drop. the improvement surcharge began being levied 

####                                in 2015

#### ***total_amount*** -- The total amount charged to passengers. Does not include cash tips.

df.shape


```python
df.dtypes
```




    VendorID                 float64
    tpep_pickup_datetime      object
    tpep_dropoff_datetime     object
    passenger_count          float64
    trip_distance            float64
    RatecodeID               float64
    store_and_fwd_flag        object
    PULocationID               int64
    DOLocationID               int64
    payment_type             float64
    fare_amount              float64
    extra                    float64
    mta_tax                  float64
    tip_amount               float64
    tolls_amount             float64
    improvement_surcharge    float64
    total_amount             float64
    congestion_surcharge     float64
    dtype: object




```python
# Calculating duration from the pickup and dropoff datetime in minutes

# converting pickup and dropoff to datetime
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

# subtracting the pickup time from dropoff time to get duration
df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']

# converting into minutes
df['duration'] = df['duration'].dt.total_seconds()/60
```


```python
df.dtypes
```




    VendorID                        float64
    tpep_pickup_datetime     datetime64[ns]
    tpep_dropoff_datetime    datetime64[ns]
    passenger_count                 float64
    trip_distance                   float64
    RatecodeID                      float64
    store_and_fwd_flag               object
    PULocationID                      int64
    DOLocationID                      int64
    payment_type                    float64
    fare_amount                     float64
    extra                           float64
    mta_tax                         float64
    tip_amount                      float64
    tolls_amount                    float64
    improvement_surcharge           float64
    total_amount                    float64
    congestion_surcharge            float64
    duration                        float64
    dtype: object




```python
df['duration'] = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
df['duration'] = df['duration'].dt.total_seconds()/60
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>congestion_surcharge</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2020-01-01 00:28:15</td>
      <td>2020-01-01 00:33:03</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>N</td>
      <td>238</td>
      <td>239</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.5</td>
      <td>1.47</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>11.27</td>
      <td>2.5</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2020-01-01 00:35:39</td>
      <td>2020-01-01 00:43:04</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>N</td>
      <td>239</td>
      <td>238</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>0.5</td>
      <td>1.50</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>12.30</td>
      <td>2.5</td>
      <td>7.416667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2020-01-01 00:47:41</td>
      <td>2020-01-01 00:53:52</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>N</td>
      <td>238</td>
      <td>238</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>10.80</td>
      <td>2.5</td>
      <td>6.183333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2020-01-01 00:55:23</td>
      <td>2020-01-01 01:00:14</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>N</td>
      <td>238</td>
      <td>151</td>
      <td>1.0</td>
      <td>5.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.36</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>8.16</td>
      <td>0.0</td>
      <td>4.850000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>2020-01-01 00:01:58</td>
      <td>2020-01-01 00:04:16</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>N</td>
      <td>193</td>
      <td>193</td>
      <td>2.0</td>
      <td>3.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.80</td>
      <td>0.0</td>
      <td>2.300000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df[["passenger_count","payment_type","fare_amount","trip_distance","duration"]]
```


```python
df.isnull().sum()
```




    passenger_count    65441
    payment_type       65441
    fare_amount            0
    trip_distance          0
    duration               0
    dtype: int64




```python
df.dropna(inplace = True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>passenger_count</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>trip_distance</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.20</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.20</td>
      <td>7.416667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0.60</td>
      <td>6.183333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.5</td>
      <td>0.80</td>
      <td>4.850000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.5</td>
      <td>0.00</td>
      <td>2.300000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6339562</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>2.10</td>
      <td>14.233333</td>
    </tr>
    <tr>
      <th>6339563</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>2.13</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>6339564</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>12.5</td>
      <td>2.55</td>
      <td>16.283333</td>
    </tr>
    <tr>
      <th>6339565</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.5</td>
      <td>1.61</td>
      <td>9.633333</td>
    </tr>
    <tr>
      <th>6339566</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.066667</td>
    </tr>
  </tbody>
</table>
<p>6339567 rows × 5 columns</p>
</div>




```python
df['passenger_count'] = df['passenger_count'].astype('int64')
df['payment_type'] = df['payment_type'].astype('int64')
```


```python
df[df.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>passenger_count</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>trip_distance</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2056</th>
      <td>1</td>
      <td>2</td>
      <td>7.0</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2441</th>
      <td>1</td>
      <td>1</td>
      <td>52.0</td>
      <td>0.00</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>2446</th>
      <td>2</td>
      <td>1</td>
      <td>9.5</td>
      <td>1.70</td>
      <td>13.066667</td>
    </tr>
    <tr>
      <th>2465</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>0.40</td>
      <td>3.083333</td>
    </tr>
    <tr>
      <th>3344</th>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>1.20</td>
      <td>5.350000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6339558</th>
      <td>1</td>
      <td>2</td>
      <td>8.0</td>
      <td>1.63</td>
      <td>8.800000</td>
    </tr>
    <tr>
      <th>6339559</th>
      <td>1</td>
      <td>1</td>
      <td>8.5</td>
      <td>1.81</td>
      <td>8.016667</td>
    </tr>
    <tr>
      <th>6339560</th>
      <td>1</td>
      <td>2</td>
      <td>6.5</td>
      <td>0.98</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>6339562</th>
      <td>1</td>
      <td>1</td>
      <td>11.0</td>
      <td>2.10</td>
      <td>14.233333</td>
    </tr>
    <tr>
      <th>6339565</th>
      <td>1</td>
      <td>2</td>
      <td>8.5</td>
      <td>1.61</td>
      <td>9.633333</td>
    </tr>
  </tbody>
</table>
<p>3331706 rows × 5 columns</p>
</div>




```python
df.drop_duplicates(inplace = True)
```


```python
df.shape
```




    (3007861, 5)




```python
df['passenger_count'].value_counts(normalize = True)
```




    passenger_count
    1    0.581981
    2    0.190350
    3    0.066360
    5    0.062937
    6    0.039272
    4    0.036046
    0    0.023033
    7    0.000009
    9    0.000006
    8    0.000006
    Name: proportion, dtype: float64




```python
df['payment_type'].value_counts(normalize = True)
```




    payment_type
    1    6.782670e-01
    2    3.075731e-01
    3    8.721480e-03
    4    5.438084e-03
    5    3.324622e-07
    Name: proportion, dtype: float64




```python
df = df[df['payment_type']<3]
df = df[(df['passenger_count'] >0)&(df['passenger_count']<6)]
```


```python
df.shape
```




    (2780283, 5)




```python
df['payment_type'].replace([1,2],['Card','Cash'],inplace = True)
```


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>passenger_count</th>
      <th>fare_amount</th>
      <th>trip_distance</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.780283e+06</td>
      <td>2.780283e+06</td>
      <td>2.780283e+06</td>
      <td>2.780283e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.733386e+00</td>
      <td>1.780567e+01</td>
      <td>4.536729e+00</td>
      <td>2.415478e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.176652e+00</td>
      <td>1.506997e+01</td>
      <td>4.895890e+00</td>
      <td>9.260031e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>-5.000000e+02</td>
      <td>-2.218000e+01</td>
      <td>-2.770367e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000e+00</td>
      <td>9.000000e+00</td>
      <td>1.500000e+00</td>
      <td>9.883333e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000e+00</td>
      <td>1.300000e+01</td>
      <td>2.730000e+00</td>
      <td>1.573333e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000e+00</td>
      <td>2.100000e+01</td>
      <td>5.470000e+00</td>
      <td>2.336667e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000e+00</td>
      <td>4.265000e+03</td>
      <td>2.628800e+02</td>
      <td>8.525117e+03</td>
    </tr>
  </tbody>
</table>
</div>



Upon reviewing the provided statistics, it's evident that the minimum values for trip distance, fare amount, and duration are negative, which is unrealistic and invalid for further analysis. Consequently, we will eliminate these negative values from the dataset.

Furthermore, observing the maximum and 50th percentile values, it's possible that the data contains significant outliers, particularly high values. These outliers need to be addressed and removed to ensure the integrity of the analysis.


```python
# filtering the records for only positive values
df = df[df['fare_amount']>0]
df = df[df['trip_distance']>0]
df = df[df['duration']>0]
```


```python
# check for the outliers
sns.boxplot(data=df, y="fare_amount", x="payment_type")
```




    <Axes: xlabel='payment_type', ylabel='fare_amount'>




    
![png](output_35_1.png)
    



```python
# removing outliers using interquartile range for the numerical variables
for col in ['fare_amount','trip_distance','duration']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3-q1

   # Define lower and upper bounds for outliers
    lower_bound = q1-1.5*IQR
    upper_bound = q3+1.5*IQR

    # Filter out outliers
    df = df[(df[col]>=lower_bound) & (df[col]<=upper_bound)]
```

We're interested on exploring the relationship between payment type and passenger behavior concerning trip distance and fare amount. Are there variations in the distribution of payment types concerning different fare amounts or trip distances?

To investigate this, we'll plot histograms to visualize the distribution of passenger counts paying with either card or cash. This will also provide stakeholders with insight into fare amount ranges associated with different payment methods.


```python
plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.title('Distribution of trip distance')
plt.hist(df[df['payment_type']=='Card']['fare_amount'], histtype= 'barstacked', bins = 20, edgecolor = 'k', color = '#A9B0EF', label = 'Card')
plt.hist(df[df['payment_type']=='Cash']['fare_amount'], histtype= 'barstacked', bins = 20, edgecolor = 'k', color = '#E9AFAF', label = 'Cash')
plt.legend()
plt.show()

plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.title('Distribution of trip distance')
plt.hist(df[df['payment_type']=='Card']['trip_distance'], histtype= 'barstacked', bins = 20, edgecolor = 'k', color = '#A9B0EF', label = 'Card')
plt.hist(df[df['payment_type']=='Cash']['trip_distance'], histtype= 'barstacked', bins = 20, edgecolor = 'k', color = '#E9AFAF', label = 'Cash')
plt.legend()
plt.show()
```


    
![png](output_38_0.png)
    



    
![png](output_38_1.png)
    



```python
# calculating the mean and standard deviation group by on payment type 
df.groupby('payment_type').agg({'fare_amount':['mean','std'], 'trip_distance':['mean','std']})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">fare_amount</th>
      <th colspan="2" halign="left">trip_distance</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>payment_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Card</th>
      <td>13.112493</td>
      <td>5.849281</td>
      <td>2.992237</td>
      <td>1.99274</td>
    </tr>
    <tr>
      <th>Cash</th>
      <td>11.758005</td>
      <td>5.613038</td>
      <td>2.602207</td>
      <td>1.91372</td>
    </tr>
  </tbody>
</table>
</div>



Now, in order to examine the passenger's preference regarding their choice of payment method, we will assess the proportion of the two payment types. To provide a visual representation, we have opted to utilize a pie chart. This graphical depiction will offer a clear and intuitive understanding of the distribution between the two payment methods chosen by passengers.


```python
plt.title('Preference of Payment Type')
plt.pie(df['payment_type'].value_counts(normalize = True), labels =df['payment_type'].value_counts().index, startangle =90,
        shadow = True, autopct = '%1.1f%%', colors = ['#A9B0EF', '#E9AFAF'])
plt.show()
```


    
![png](output_41_0.png)
    



```python
# calculating the total passenger count distribution based on the different payment type
passenger_count = df.groupby(['payment_type','passenger_count'])[['passenger_count']].count()

# renaming the passenger_count to count to reset the index
passenger_count.rename(columns = {'passenger_count':'count'}, inplace = True)
passenger_count.reset_index(inplace = True)
```


```python
# calculating the percentage of the each passenger count
passenger_count['perc'] = (passenger_count['count']/passenger_count['count'].sum())*100
```


```python
passenger_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>payment_type</th>
      <th>passenger_count</th>
      <th>count</th>
      <th>perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Card</td>
      <td>1</td>
      <td>909245</td>
      <td>39.568381</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Card</td>
      <td>2</td>
      <td>327661</td>
      <td>14.259100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Card</td>
      <td>3</td>
      <td>122412</td>
      <td>5.327106</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Card</td>
      <td>4</td>
      <td>63676</td>
      <td>2.771042</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Card</td>
      <td>5</td>
      <td>124045</td>
      <td>5.398171</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cash</td>
      <td>1</td>
      <td>460550</td>
      <td>20.042143</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cash</td>
      <td>2</td>
      <td>155472</td>
      <td>6.765806</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cash</td>
      <td>3</td>
      <td>54506</td>
      <td>2.371984</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cash</td>
      <td>4</td>
      <td>32715</td>
      <td>1.423686</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cash</td>
      <td>5</td>
      <td>47626</td>
      <td>2.072581</td>
    </tr>
  </tbody>
</table>
</div>




```python
# creating a new empty dataframe to store the distribution of each payment type (useful for the visualization)
df1 = pd.DataFrame(columns = ['payment_type',1,2,3,4,5])
df1['payment_type'] = ['Card', 'Cash']
df1.iloc[0,1:] = passenger_count.iloc[0:5,-1]
df1.iloc[1,1:] = passenger_count.iloc[5:,-1]
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>payment_type</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Card</td>
      <td>39.568381</td>
      <td>14.2591</td>
      <td>5.327106</td>
      <td>2.771042</td>
      <td>5.398171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cash</td>
      <td>20.042143</td>
      <td>6.765806</td>
      <td>2.371984</td>
      <td>1.423686</td>
      <td>2.072581</td>
    </tr>
  </tbody>
</table>
</div>



Subsequently, we aim to conduct an analysis of the payment types in relation to the passenger count. Our objective is to investigate if there are any changes in preference contingent upon the number of passengers traveling in the cab.

To facilitate this examination, we have employed a visualization technique known as a stacked bar plot. This method is particularly advantageous for comparing the percentage distribution of each passenger count based on the payment method selected. Through this graphical representation, we can gain insights into potential variations in payment preferences across different passenger counts.


```python
fig, ax = plt.subplots(figsize=(20, 6))
df1.plot(x= 'payment_type', kind = 'barh', stacked = True, ax= ax, color = ['#A9B0EF', '#E9AFAF', '#CBB2B2', '#F1F1F1', '#FD9F9F'])

#Add percentage text
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2,
            y + height / 2,
            '{:0.00f}%'.format(width),
            horizontalalignment='center',
            verticalalignment='center')
```


    
![png](output_47_0.png)
    


# Hypothesis Testing

**Null hypothesis**: There is no difference in average fare between customers who use credit cards and customers who use cash.

**Alternative hypothesis**: There is a difference in average fare between customers who use credit cards and customers who use cash.

In order to select the most suitable test for our scenario, our initial step involves evaluating whether the distribution of fare amounts adheres to a normal distribution. While the histogram depicted above suggests otherwise, we will further confirm this by generating a QQ plot.

Quantile-quantile (QQ) plots can be used to assess whether the fare amount distributions for each payment type are approximately normally distributed. If the data points closely align with the diagonal line in the plot, it suggests that the data follows a normal distribution.


```python
import statsmodels.api as sm
```


```python
#create Q-Q plot with 45-degree line added to plot
sm.qqplot(df['fare_amount'], line = '45')
plt.show()
```


    
![png](output_52_0.png)
    


The data values clearly do not follow the red 45-degree line, which is an indication that they do not follow a normal distribution. So, Z-Test will not be good for this. That's why we will use T test.

Given that the T-test can be applied to both small and large samples and does not require the population standard deviation, it is a more universally applicable approach for hypothesis testing in many practical research scenarios, including analyses of taxi trip data.

In the analysis of NYC Yellow Taxi Trip Records, where you're likely dealing with an unknown population standard deviation and potentially large datasets, the T-test offers a more appropriate and flexible method for comparing means between two groups (e.g., fare amounts by payment type). It provides a reliable way to infer about the population, accommodating the uncertainty that comes with estimating population parameters from sample data.


```python
# sample 1
card_sample = df[df['payment_type']=='Card']['fare_amount']
# sample 2
cash_sample = df[df['payment_type']=='Cash']['fare_amount']
```


```python
# performing t test on both the different sample
t_stats, p_value = st.ttest_ind(a = card_sample, b = cash_sample, equal_var = False)
print('T statistic', t_stats, 'p_value',p_value)

# comparing the p value with the significance of 5% or 0.05
if p_value < 0.05:
    print("\nReject the null hypothesis")
else:
    print("\nAccept the null hypothesis")
```

    T statistic 169.2111527245052 p_value 0.0
    
    Reject the null hypothesis
    

Since the p-value is significantly smaller than the significance level of 5%, we will reject the null hypothesis.

You conclude that there is a statistically significant difference in the average fare amount between customers who use credit cards and customers who use cash.

The key business insight is that encouraging customers to pay with credit cards can generate more revenue for taxi cab drivers.

# Recommendations

Encourage customers to pay with credit cards to capitalize on the potential forgenerating more revenue for taxi cab drivers.

Implement strategies such as offering incentives or discounts for credit cardtransactions to incentivize customers to choose this payment method.

Encourage customers to pay with credit cards capitalize on the potential for generating more revenue taxi cab drivers. 
