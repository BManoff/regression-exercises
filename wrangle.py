from pydataset import data # importing librabries
import seaborn as sns
import pandas as pd
import os
import env
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

# connect to the mysql server
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_wrangle_zillow():
    # Get local cached file if it's there
    filename = "zillow.csv" 

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql(
        ''' 
        SELECT

        bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips

        FROM 

        properties_2017 

        JOIN 
        
        propertylandusetype using (propertyusetypeid)

        WHERE
         
        propertylandusedesc = "Single Family Residential"

        '''
        
        , get_connection('zillow')
        )


        # Return the dataframe to the calling code
        return df  

def handle_nulls(df):
      
      df = df.dropna() #dropping all the na values

      return df


def type_change(df):
    df["fips"] = df["fips"].astype(int)

    df["yearbuilt"] = df["yearbuilt"].astype(int)

    df["bedroomcnt"] = df["bedroomcnt"].astype(int)   

    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)

    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)

    return df

def handle_outliers(df):
    
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    return df

def wrangle_zillow():

   df = get_wrangle_zillow()

   df = handle_nulls(df)

   df = type_change(df)

   df = handle_outliers(df)

   df.to_csv('zillow.csv', index=False)

   return df

def scale_data(train, validate, test, return_scaler=False):
  
    columns_to_scale = ['bedrooms', 'bathrooms', 'tax_value', 'taxamount', 'area']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
