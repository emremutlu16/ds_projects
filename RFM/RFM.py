###############################################################
# Data Understanding
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")


df = df_.copy()
df.head()
df.info()
df.isnull().sum()


# unique product number
df["Description"].nunique()

# counts of every product
df["Description"].value_counts().head()

# most ordered product
df.groupby("Description").agg({"Quantity": "sum"}).head()
df.groupby("Description").agg({"Quantity": "sum"}).\
    sort_values("Quantity", ascending=False).head()

# how many invoices have been made
df["Invoice"].nunique()

# average earning over one invoice
# "C" means invoice is canceled so lets take these out
df = df[~df["Invoice"].str.contains("C", na=False)]
df["TotalPrice"] = df["Quantity"] * df["Price"]

# most expensive products
df.sort_values("Price", ascending=False).head()

# How many orders have arrived from which country?
df["Country"].value_counts()

# Which country made how much?
df.groupby("Country").agg({"TotalPrice": "sum"}).\
    sort_values("TotalPrice", ascending=False).head()



###############################################################
# Data Preparation
###############################################################

df.isnull().sum()
df.dropna(inplace=True)

df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T


###############################################################
# Calculating RFM Metrics
###############################################################

# setting a date by adding two days to the maximum date on the invoice to
# apply recency to data
df["InvoiceDate"].max()

today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda date: (today_date - date.max()).days,
    'Invoice': lambda num: len(num),
    'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ['Recency', 'Frequency', 'Monetary']


# extraction of data without monetary and frequency
rfm = rfm[(rfm["Monetary"]) > 0 & (rfm["Frequency"] > 0)]


###############################################################
# Calculating RFM Scores
###############################################################


# Recency
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])

rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str))

rfm[rfm["RFM_SCORE"] == "555"].head()

rfm[rfm["RFM_SCORE"] == "111"]

###############################################################
# Naming & Analysing RFM Segments
###############################################################

# RFM naming
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'
}

# merge without monetary to segment by regex
rfm['Segment'] = rfm['RecencyScore'].astype(str) + \
                 rfm['FrequencyScore'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
df[["Customer ID"]].nunique()
rfm[["Segment", "Recency", "Frequency", "Monetary"]].\
    groupby("Segment").agg(["mean", "count"])

# Information about how many customers are in segments
rfm['Segment'].value_counts()


rfm[rfm["Segment"] == "Need_Attention"].head()
rfm[rfm["Segment"] == "Need_Attention"].index


# creating excel for loyal customers
new_df = pd.DataFrame()

new_df["Loyal_Customers"] = rfm[rfm["Segment"] == "Loyal_Customers"].index

new_df.to_excel("Loyal_Customers.xlsx")
