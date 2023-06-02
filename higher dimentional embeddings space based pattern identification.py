import requests
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding as openai_get_embedding
from openai.embeddings_utils import cosine_similarity
from matplotlib import pyplot as plt
import openai

# Set up OpenAI API credentials
openai.api_key = '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'

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


def fetch_price_data(timeframe, symbol, limit:str=1000):
    # Set up API endpoint
    url = 'https://api.binance.com/api/v3/klines'

    # Set up query parameters
    params = {
        'symbol': symbol,
        'interval': timeframe,
        'limit': limit  # Maximum limit for fetching historical data
    }

    # Fetch historical price data
    response = requests.get(url, params=params)
    data = response.json()


    # Convert data to pandas dataframe
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    return df[['close']]

def get_consecutive_close_prices(df, window:int):
    rows = []
    for i in range(len(df)-window+1):
        row = ','.join(map(str, df.iloc[i:i+window]['close'].tolist()))
        rows.append(row)
    return pd.DataFrame(rows, columns=['to_encode'])

def get_embedding(df): # the df should have the column to get embedded as to_encode
    df["embedding"] = df["to_encode"].apply(lambda x: openai_get_embedding(x, engine=embedding_model))
    df.to_csv("embeddings.csv")
    return df

def embed_price_data(timeframe, symbol, limit, window):
    data=fetch_price_data(timeframe, symbol, limit)
    df=get_consecutive_close_prices(data, window)
    return get_embedding(df)


def process_new_data(new_data):
    output=''
    dataList=[]
    for i in range(len(new_data)):
        output=output+str(new_data['close'].iloc[i])+','
        dataList.append(float(new_data['close'].iloc[i]))
    return output[:-1], dataList

def get_similarity(data, new_data):
    new_data_vector = openai_get_embedding(new_data, engine=embedding_model)
    data["similarities"] = data['embedding'].apply(lambda x: cosine_similarity(x, new_data_vector))
    data = data.sort_values("similarities", ascending=False).head(4)
    data.to_csv("sorted_output.csv")
    return data

def find_top_similar_pattern(new_data, df):
    n_data, data_list=process_new_data(new_data)
    sorted=get_similarity(df, n_data)

    return data_list, sorted


df = pd.read_csv('Binance_BTCUSDT_2020_minute.csv')

# flip the rows
df= df.iloc[::-1].reset_index(drop=True)

df = df['close'].pct_change() * 100

print(df.head())
#df = get_consecutive_close_prices(df, 50)
#get_embedding(df)


'''
#df=embed_price_data('15m', 'BTCUSDT', 1000, 50)
df = pd.read_csv('embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

new_data = fetch_price_data('5m','BTCUSDT', 20)
original, df = find_top_similar_pattern(new_data, df)

print(df["similarities"]*100)
df=df['to_encode']
closes = df.reset_index(drop=True)

data_list1 = [float(x) for x in closes.iloc[0].split(",")]
data_list2 = [float(x) for x in closes.iloc[1].split(",")]
data_list3 = [float(x) for x in closes.iloc[2].split(",")]
data_list4 = [float(x) for x in closes.iloc[3].split(",")]

forcast=[]
for i in range(len(data_list1)):
    forcast.append((data_list1[i]+data_list2[i]+data_list3[i]+data_list4[i])/4)


print(original)
print(data_list1)
print(data_list2)
print(data_list3)
print(data_list4)


# Plotting the four lists
plt.plot(original, label='OG', c='blue')
plt.plot(forcast, label='forcast', c='gray')
plt.plot(data_list1, label='List 1', c='green')
plt.plot(data_list2, label='List 2', c='black')
plt.plot(data_list3, label='List 3', c='red')
plt.plot(data_list4, label='List 4', c='yellow')
# Showing the plot
plt.show()
'''


