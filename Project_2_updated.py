#!/usr/bin/env python
# coding: utf-8

# ### Before starting the project , i want really thank `EPSILON` for their efforts 
# I started my journey with them about a 9 month ago , i was very intesrested with my new journy as a step by step data scienest , I started my round with **ENG/Ahmed Noaman** and **ENG/Ayed Ali** but unfortunately I experienced some health conditions leading me to do two surgeries and making a freeze for the diploma , after I got well and come back from the freeze i met agreat instructor **ENG/Salah Tarek** who i want to realy thank him for his effort and of course **ENG/Mohab Allam** , today el hamdullah i am with good health condithions , i have graduated from my college with excellent degree , witnissing the end of the journey with epsilon but it is not an end it is just a start with another journey with them , Now i am doing an end to end data science project :) 

# ## LifeCycle of our project 
# - Unterstanding the Problem Statemet
# - Data Colleection
# - Data Cleaning Phase
# - Explotary Data Analysis (EDA)
# - Feature Engineering
# - EDA & Data Visualization
# - data Preprocessing (DPP)
# - Feature Selection
# - Pick and Tune an Algorithm
# - Validate and Evaluate
# - Best Model Selection
# - Project Deployment

# ### Problem statement
# - In this dataset we will predict the price of any property in egypt using regression models , we have a good dataset , our dataset has 27361 values(records) and 12 columns(features). We will prepare our data and use machine learning models to achieve our goal.
# 
# 
# ### Data Collection
# - Dataset Source - https://www.kaggle.com/datasets/mohammedaltet/egypt-houses-price/data
# - The data set consists of 12 column (our main features).

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Egypt_Houses_Price.csv', sep = ',')
df.head()


# In[3]:


df.info()


# **Insights**
# - from the info we can see that we have problem with data type and we need to work on it
# - we have some Null to deal whith

# ## Data Cleaning Phase

# **Dealing With Null And Duplicated Rows**

# In[4]:


df.isnull().sum()


# In[5]:


for x in df.columns :
    print(f"{x} has {df[x].unique()}")
    print("*"*20)


# we have many unkowns

# In[6]:


df['Area'] = df['Area'].replace('Unknown', np.nan)
df['Bedrooms'] = df['Bedrooms'].replace('Unknown', np.nan)
df['Bathrooms'] = df['Bathrooms'].replace('Unknown', np.nan)
df['Price'] = df['Price'].replace('Unknown', np.nan)
#changing Unkhown data to NAN 


# In[7]:


df.isnull().sum()


# So we can drop null values as it is not too much

# In[8]:


df.dropna(inplace=True)


# In[9]:


df.duplicated().sum()


# In[10]:


df.drop_duplicates(inplace=True)


# ### Data Preprocessing (DPP)

# ### Fixing the Dtype for the columns

# In[11]:


df.info()


# In[12]:


for x in df.columns :
    print(f"{x} has {df[x].unique()}")
    print("*"*20)


# In[13]:


df['Bedrooms'] = df['Bedrooms'].replace('10+',11)
df['Bathrooms'] = df['Bathrooms'].replace('10+',11)


# In[14]:


df['Bedrooms'] = df['Bedrooms'].astype(float).astype(int)
df['Bathrooms'] = df['Bathrooms'].astype(float).astype(int)
df['Area'] = df['Area'].astype(float).astype(int)
df['Price'] = df['Price'].astype(int)


# In[15]:


print(df['Type'].unique())
print(df['Level'].unique())


# ### From searching info we got that :
# - ( Duplex , Apartment , Studio ) type can be in different level 
# - ( Twin house , Town House , Stand Alone Villa ,Chalet ) type only on Ground level
# - ( Penthouse ) type is on Highest only

# ### From type of property , we will change the level from int to str 
# - Ground = 0
# - 10+ = 11
# - Highest = 12

# In[16]:


df=df.drop(df[(df['Level']=='Unknown')&(df['Type']=='Duplex')].index)
df=df.drop(df[(df['Level']=='Unknown')&(df['Type']=='Apartment')].index)
df=df.drop(df[(df['Level']=='Unknown')&(df['Type']=='Studio')].index)


# In[17]:


df.loc[(df['Level']=='10+'),'Level'] = 11
df.loc[(df['Level']=='Highest'),'Level'] = 12
df.loc[(df['Level']=='Ground'),'Level'] = 0


# In[18]:


df.loc[(df['Type']=='Penthouse')|
       (df['Type']=='Standalone Villa')|
       (df['Type']=='Town House')|
       (df['Type']=='Twin house')|(df['Type']=='Stand Alone Villa')|
       (df['Type']=='Chalet')|
       (df['Type']=='Twin House'),'Level'] = 0
df.loc[(df['Type']=='Penthouse'),'Level'] = 12


# In[19]:


df['Level'] = df['Level'].astype(float).astype(int)


# ### Furnished Column

# In[20]:


df['Delivery_Date'].unique()


# In[21]:


df[(df['Furnished']=='Unknown')&(df['Delivery_Date']!='Ready to move')&(df['Delivery_Date']!='Unknown')]
#we can replace the Furnished data here naturally with NO


# In[22]:


df.loc[(df['Furnished']=='Unknown')&(df['Delivery_Date']!='Ready to move')&(df['Delivery_Date']!='Unknown'),'Furnished'] = 'No'


# In[23]:


df['Delivery_Term'].unique()


# In[24]:


df.loc[(df['Furnished']=='Unknown')&(df['Delivery_Term']!='Finished')&(df['Delivery_Term']!='Unknown ')]
# We can replace the Furnished data here naturally with NO bec the Delicery Term is not finished yet


# In[25]:


df.loc[(df['Furnished']=='Unknown')&(df['Delivery_Term']!='Finished')&(df['Delivery_Term']!='Unknown '),'Furnished'] = 'No'


# ### Drop columns that have more than 30% of NANs

# In[26]:


furnished = len(df[df['Furnished'] == 'Unknown']) / len(df)
level = len(df[df['Level'] == 'Unknown']) / len(df)
compound = len(df[df['Compound'] == 'Unknown']) / len(df)
Payment_Option = len(df[df['Payment_Option'] == 'Unknown']) / len(df)
Delivery_Date = len(df[df['Delivery_Date'] == 'Unknown']) / len(df)
Delivery_Term = len(df[df['Delivery_Term'] == 'Unknown ']) / len(df)
City = len(df[df['City'] == 'Unknown']) / len(df)


# In[27]:


print('Furnished: ', furnished)
print('Level: ',level)
print('Compound: ',compound)
print('Payment_Option: ',Payment_Option)
print('Delivery_Date: ',Delivery_Date)
print('Delivery_Term: ',Delivery_Term)
print('City: ',City)


# In[28]:


df = df.drop('Compound',1)
df = df.drop('Delivery_Date',1)


# In[29]:


df.head()


# In[30]:


df['Furnished'] = df['Furnished'].replace('Unknown', np.nan)
df['Delivery_Term'] = df['Delivery_Term'].replace('Unknown ', np.nan)
df.dropna(inplace=True)


# In[31]:


df.reset_index(inplace=True)
df.drop(['index'],axis=1,inplace=True)


# In[32]:


df.info()


# In[33]:


df.isnull().sum()


# In[34]:


df.describe().T


# In[35]:


df.describe(include=['O']).T


# ### Type Column

# In[36]:


df['Type'].unique()


# In[37]:


df['Type'].value_counts()


# In[38]:


df.loc[(df['Type']=='Standalone Villa'),'Type'] = 'Stand Alone Villa'
df.loc[(df['Type']=='Twin house'),'Type'] = 'Twin House'


# In[39]:


df['Type'].value_counts()


# ### City Column

# #### Deleting Location/City that have less than 5 rows
# - city with low data don't give enough help to the model it's better to drop it

# In[40]:


pd.set_option('display.max_rows', 500)
df['City'].value_counts(ascending=True)


# In[41]:


ind = df['City'].value_counts(dropna=False).keys().tolist()
val = df['City'].value_counts(dropna=False).tolist()
value_dict = list(zip(ind, val))


# In[42]:


value_dict


# In[43]:


lc_sm = []
y = 'Less'
for val,ind in value_dict:
    if ind <= 5:
        lc_sm.append(val)
    else :
        pass
def lcdlt(x):
    if x in lc_sm:
        return y
    else :
        return x


# In[44]:


df['City'] = df['City'].apply(lcdlt)


# In[45]:


df=df.drop(df[(df['City']=='Less')].index)


# ### price depends highly with the location so we will deduct outlines for every city manualy

# In[46]:


plt.subplots(figsize=(24, 128))
sns.boxplot(y='City', x='Price',data=df);


# In[47]:


lcc = df['City'].value_counts().keys().tolist()


# In[48]:


for x in lcc:
    Q1= df[(df['City']==x)]['Price'].quantile(0.25)
    Q3= df[(df['City']==x)]['Price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.2 * IQR
    lower_bound = Q1 - 1.2 * IQR
    df=df.drop(df[(df['City']==x)&(df['Price']>=upper_bound)].index)
    df=df.drop(df[(df['City']==x)&(df['Price']<=lower_bound)].index)


# In[49]:


df['City'].unique()


# In[50]:


df=df.drop(df[df['City']=='(View phone number)'].index)


# In[51]:


df.reset_index(inplace=True)
df.drop(['index'],axis=1,inplace=True)


# ### Area Column

# In[52]:


df['Area'].describe()


# In[53]:


#deleting some raws that doesn't make sense like 4 rooms in 100 meters
df=df.drop(df[(df['Area']<=100)&(df['Bedrooms']>=4)].index)
df=df.drop(df[(df['Area']<=30)&(df['Type']!='Studio')].index)


# In[54]:


df[(df['Area']>=300)&(df['Price']<=2000000)&(df['Payment_Option']=='Cash')&(df['Delivery_Term']=='Finished')]
df=df.drop(df[(df['Area']>=300)&(df['Price']<=2000000)&(df['Payment_Option']=='Cash')&(df['Delivery_Term']=='Finished')].index)


# ### Payment_Option Column
# 

# In[55]:


def Price_range(x) :
    if x <= 1000000 : 
        return 'Low Price'
    elif x <= 3000000 :
        return 'Mid Price'
    else :
        return 'high Price'


# In[56]:


df['Price_range'] = df['Price'].apply(Price_range)


# In[57]:


df.groupby('Price_range')['Payment_Option'].value_counts()


# ### Visualization

# In[58]:


for col in df.columns:
    print(col,':',df[col].nunique())
    print(df[col].value_counts().nlargest(7))
    print('\n'+'*'*20+'\n')


# In[59]:


mp = df['City'].value_counts()[0:20].sort_values()
sns.barplot(y=mp.index,x=mp.values);
plt.title('The City With The Most Buildings ');


# In[60]:


lpm = df.groupby('City')['Price'].mean()[0:20].sort_values()
sns.barplot(y=lpm.index,x=lpm.values);
plt.title('The City With The Higher Buildings Price');


# In[61]:


lpp = df.groupby('Type')['Price'].mean().sort_values()
sns.barplot(y=lpp.index,x=lpp.values);
plt.title('Most Expensive Property Type Building');


# In[62]:


lpb = df.groupby('Type')['Bedrooms'].mean().sort_values()
sns.barplot(y=lpb.index,x=lpb.values);
plt.title('Property With Mean Bed Room Number');


# In[63]:


df['Furnished'].value_counts().plot(kind='pie');


# In[64]:


df.groupby('Furnished')['Price'].mean().sort_values().plot(kind='pie');


# In[65]:


ind=(df.groupby('City')['Price'].sum()/df.groupby('City')['Area'].sum()).sort_values(ascending=False)[0:30].index
vlu=(df.groupby('City')['Price'].sum()/df.groupby('City')['Area'].sum()).sort_values(ascending=False)[0:30].values
sns.barplot(data = df, x = ind ,y= vlu ,ci = None,order = ind);
plt.xticks(rotation=90);
plt.title('The City With The 1 Meter^2 Price');


# # From Visualization we can say the most important points:
# - Price depend on the city highly than on space of the property.
# - Furnished or Not Furnished does not affect highly on Price.

# In[66]:


sns.pairplot(df, vars = ['Price', 'Area'], height=5, aspect=1.3);


# In[67]:


sns.pairplot(df, vars = ['Price', 'Level'], height=5, aspect=1.3);


# In[68]:


df.hist(bins=50, figsize=(15, 15));


# In[69]:


sns.histplot(data=df,x='Price',bins=150);


# In[70]:


sns.boxplot(x='Area', data=df)
sns.stripplot(x='Area', data=df, color="#474646");


# In[71]:


sns.histplot(data = df, x ='Area', kde = True)


# In[72]:


df.corr()


# In[73]:


df.to_csv('New_houes_data.csv')


# ### Now let's start our machine learning phase  

# In[74]:


df = pd.get_dummies(df, columns = ['Type', 'Furnished','City' ,'Payment_Option','Delivery_Term'])
X = df.drop(columns = ['Price','Price_range'])
y = df[['Price']]


# In[75]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.25,shuffle = True ,random_state = 404)

X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[76]:


from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[77]:


models = {
    "LR": LinearRegression(),
    "KNNR" : KNeighborsRegressor(), 
    "Lasso":Lasso(),
    "SVR": SVR(),
    "DT": DecisionTreeRegressor(),
    "RF": RandomForestRegressor(),
    "XGBR": XGBRegressor()
}


# ### Validate and Evaluate

# In[78]:


for name, model in models.items():
    print(f'Using model: {name}')
    model.fit(X_train, y_train)
    print(f'Training Score: {model.score(X_train, y_train)}')
    print(f'Test Score: {model.score(X_test, y_test)}')
    y_pred = model.predict(X_test)
    print(f'MSE Score: {np.sqrt(mean_squared_error(y_test, y_pred))}')  
    print(f'R2 Score: {r2_score(y_test, y_pred)}') 
    print('-'*30)


# ### Best model selection

# In[79]:


results = []

for name, model in models.items():
    print(f'Using model: {name}')
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Train Score': train_score,
        'Test Score': test_score,
        'RMSE': rmse,
        'R2': r2
    })
    
    print(f'Training Score: {train_score}')
    print(f'Test Score: {test_score}')
    print(f'RMSE: {rmse}')
    print(f'R2 Score: {r2}')
    print('-' * 30)

results_df = pd.DataFrame(results)

results_df_sorted = results_df.sort_values('Test Score', ascending=False)

print("\n=== Models Ranked by Test Score (Descending) ===")
print(results_df_sorted)


# ### Tuning and optimization

# In[80]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[81]:


def performance(model,X_train,y_train,y_pred,y_test):
    '''
    This function for calculating the performance of the model.
    INPUT:
    model: Regression Model. The regression model.
    x_train: numpy.ndarray. The training data in the x.
    y_train: pandas.core.series.Series. The training data in the y.
    y_pred: numpy.ndarray. The predicted data.
    y_test: pandas.core.series.Series. The actual data.
    OUTPUT:
    The model performance by different metrics.
    '''
    print('Training Score:',model.score(X_train,y_train))
    print('Testing Score:',r2_score(y_test,y_pred))
    print('Other Metrics In Testing Data: ')
    print('MSE:',mean_squared_error(y_test,y_pred))
    print('MAE:',mean_absolute_error(y_test,y_pred))


# ### RandomForest

# In[82]:


#Randomized search
params = [
    {'n_estimators':[100,200,3000,400,500,600],
     'max_depth':list(range(5,20)),'min_samples_split':list(range(2,15)),"min_samples_leaf":[2,3,4,5]
     }
         ]
rand_search = RandomizedSearchCV(RandomForestRegressor(),params,cv=10,n_jobs=-1)

#Fitting the model
rand_search.fit(X_train,y_train.values.ravel())

#The best estimator
print('Best Estimator:',rand_search.best_estimator_)

#The best parameters
print('Best Params:',rand_search.best_params_)

#The predicted data
rand_pred = rand_search.predict(X_test)

#Decision tree performance after tuning
performance(rand_search,X_train,y_train,rand_pred,y_test)

#Plotting the results
plt.scatter(rand_pred,y_test,c='blue',marker='o',s=25)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],c='black',lw=2)
plt.xlabel('Predicted Data',c='red')
plt.ylabel('Actual Data',c='red')
plt.title('Predicted Data VS Actual Data',c='red')
plt.show()


# ### XGBRegressor

# In[83]:


#Randomized search
params = {
         'max_depth': list(range(5,15)),'n_estimators': [300,400,500,600,700],'learning_rate': [0.01,0.1,0.2,0.9]
         }
rand_search = RandomizedSearchCV(XGBRegressor(),params,cv=10,n_jobs=-1)

#Fitting the model
rand_search.fit(X_train,y_train)

#The best estimator
print('Best Estimator:',rand_search.best_estimator_)

#The best parameters
print('Best Params:',rand_search.best_params_)

#The predicted data
rand_pred = rand_search.predict(X_test)

#Decision tree performance after tuning
performance(rand_search,X_train,y_train,rand_pred,y_test)

#Plotting the results
plt.scatter(rand_pred,y_test,c='blue',marker='o',s=25)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],c='black',lw=4)
plt.xlabel('Predicted Data',c='red')
plt.ylabel('Actual Data',c='red')
plt.title('Predicted Data VS Actual Data',c='red')
plt.show()


# in the end we get mid result using XGBRegressor model
# with 
# - Training Score: 0.8383016048773416
# - Testing Score: 0.7224066464741059

# In[84]:


#Fitting the model
from sklearn.ensemble import RandomForestRegressor

model = XGBRegressor(n_estimators=600,max_depth=9,learning_rate=0.01,random_state=42)
model.fit(X_train,y_train)

#The predicted data
model_pred = model.predict(X_test)

#The performance
performance(model,X_train,y_train,model_pred,y_test)


# In[86]:


y_predict = model.predict(X_test)
r2_Score = r2_score(y_test, y_predict)*100
print(f"accuracy is {round(r2_Score,2)}%")


# In[87]:


plt.scatter(y_test,y_predict)
plt.xlabel("True value")
plt.ylabel("Predicted value")


# In[88]:


sns.regplot(x=y_test, y=y_predict,ci=None ,color='blue')


# In[91]:


y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Prediction')

plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Actual vs Predicted Values (XGBoost Regression)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()


# ### Save the model

# In[97]:


import joblib
import pandas as pd

joblib.dump(model, 'egypt_house_price_xgboost.joblib') 
feature_names = X_train.columns.tolist()  
pd.to_pickle(feature_names, 'feature_names.pkl') 

print("Model and feature names saved successfully!")


# ### Project Deployment

# In[108]:


get_ipython().run_cell_magic('writefile', 'app.py', '\nimport streamlit as st\nimport pandas as pd\nimport joblib\nimport pickle\nfrom xgboost import XGBRegressor\n\n# Set page config - Wide mode with centered title\nst.set_page_config(\n    page_title="Egypt House Price Predictor",\n    layout="wide",\n    page_icon="🏠"\n)\n\n# Custom CSS for better styling\nst.markdown("""\n<style>\n    .stSelectbox, .stNumberInput, .stRadio > div {\n        padding: 10px !important;\n        border-radius: 8px !important;\n        border: 1px solid #e1e4e8 !important;\n    }\n    .stButton button {\n        width: 100%;\n        padding: 10px !important;\n        border-radius: 8px !important;\n        background-color: #4CAF50 !important;\n        color: white !important;\n        font-weight: bold !important;\n    }\n    .stMetric {\n        padding: 15px;\n        border-radius: 10px;\n        background-color: #f0f2f6;\n    }\n    .property-card {\n        padding: 20px;\n        border-radius: 10px;\n        box-shadow: 0 4px 8px rgba(0,0,0,0.1);\n        margin-bottom: 20px;\n    }\n</style>\n""", unsafe_allow_html=True)\n\n# Load model and features\n@st.cache_resource\ndef load_model():\n    model = joblib.load(\'egypt_house_price_xgboost.joblib\')\n    feature_names = pickle.load(open(\'feature_names.pkl\', \'rb\'))\n    return model, feature_names\n\nmodel, feature_names = load_model()\n\n# All possible values\nPROPERTY_TYPES = [\n    \'Apartment\', \'Stand Alone Villa\', \'Duplex\', \n    \'Town House\', \'Twin House\', \'Penthouse\', \n    \'Studio\', \'Chalet\'\n]\n\nCITIES = [\n    \'New Cairo\', \'6th of October\', \'Sheikh Zayed City\', \'Nasr City\',\n    \'Maadi\', \'Zamalek\', \'Heliopolis\', \'Giza\', \'Madinaty\', \'Shorouk City\',\n    \'5th Settlement\', \'Agouza\', \'Dokki\', \'Haram\', \'Mohandessin\',\n    \'El Rehab\', \'Obour City\', \'Sheraton\', \'El Tagamo3\', \'Katameya\',\n    \'Badr City\', \'Gesr El Suez\', \'Hadayek El Ahram\', \'Mostakbal City\',\n    \'Zayed\', \'El Marg\', \'10th of Ramadan\', \'15th of May\', \'Abbassia\',\n    \'Ain Shams\', \'El Manial\', \'El Mohandessin\', \'El Sakkakini\',\n    \'El Sayeda Zeinab\', \'El Shorouk\', \'Garden City\', \'Hadayek El Kobba\',\n    \'Helmeyat El Zaytoun\', \'Mokattam\', \'Rod El Farag\', \'Saft El Laban\',\n    \'Sharabi\', \'Sheraton Heliopolis\', \'Shubra\', \'Zeitoun\'\n]\n\n# App Header\nst.title("Egypt Property Price Predictor")\nst.markdown("""\n<div style=\'background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:30px;\'>\n    <h3 style=\'color:#2e86de; margin-top:0;\'>Get instant price estimates for properties across Egypt</h3>\n    <p>Our AI model analyzes market trends to provide accurate valuations based on your property details.</p>\n</div>\n""", unsafe_allow_html=True)\n\n# Main layout\ncol1, col2 = st.columns([1, 2], gap="large")\n\n# Input Form in left column\nwith col1:\n    with st.container():\n        st.subheader("Property Details")\n        \n        with st.form("property_form"):\n            # Property Characteristics\n            st.markdown("**Basic Information**")\n            property_type = st.selectbox("Type", PROPERTY_TYPES)\n            city = st.selectbox("City", sorted(CITIES))\n            area = st.number_input("Area (sqm)", min_value=30, max_value=2000, value=120)\n            \n            # Property Features\n            st.markdown("**Features**")\n            bedrooms = st.selectbox("Bedrooms", options=range(1, 12), index=2)\n            bathrooms = st.selectbox("Bathrooms", options=range(1, 12), index=1)\n            level = st.selectbox("Floor Level", \n                              options=[("Ground", 0)] + [(str(i), i) for i in range(1, 11)] + [("10+", 11), ("Penthouse (Highest)", 12)],\n                              format_func=lambda x: x[0],\n                              index=1)\n            \n            # Additional Details\n            st.markdown("**Additional Details**")\n            furnished = st.radio("Furnished", ["Yes", "No"], horizontal=True)\n            payment = st.radio("Payment Option", ["Cash", "Installment"], horizontal=True)\n            delivery = st.radio("Delivery Term", ["Finished", "Not Finished"], horizontal=True)\n            \n            # Submit button\n            submitted = st.form_submit_button("Calculate Price", type="primary")\n\n# Results in right column\nwith col2:\n    if submitted:\n        # Prepare input data\n        input_data = {\n            \'Area\': area,\n            \'Bedrooms\': bedrooms,\n            \'Bathrooms\': bathrooms,\n            \'Level\': level[1],\n            \'Type\': property_type,\n            \'City\': city,\n            \'Furnished\': furnished,\n            \'Payment_Option\': payment,\n            \'Delivery_Term\': delivery\n        }\n        \n        # Create DataFrame and one-hot encode\n        input_df = pd.DataFrame([input_data])\n        for col in [\'Type\', \'Furnished\', \'City\', \'Payment_Option\', \'Delivery_Term\']:\n            input_df[f"{col}_{input_data[col]}"] = 1\n        \n        # Ensure all feature columns exist\n        for feature in feature_names:\n            if feature not in input_df.columns:\n                input_df[feature] = 0\n        \n        # Reorder columns\n        input_df = input_df[feature_names]\n        \n        # Make prediction\n        prediction = model.predict(input_df)[0]\n        price_per_sqm = prediction / area\n        \n        # Display results\n        st.subheader("Price Estimation")\n        \n        # Main price card\n        st.markdown(f"""\n        <div class=\'property-card\' style=\'background-color:#f8f9fa;\'>\n            <div style=\'font-size:24px; color:#2e86de; font-weight:bold; margin-bottom:10px;\'>\n                Estimated Property Value\n            </div>\n            <div style=\'font-size:36px; color:#e74c3c; font-weight:bold; margin-bottom:15px;\'>\n                EGP {prediction:,.0f}\n            </div>\n            <div style=\'font-size:18px; color:#27ae60;\'>\n                Price per sqm: EGP {price_per_sqm:,.0f}\n            </div>\n        </div>\n        """, unsafe_allow_html=True)\n        \n        # Property details card\n        st.markdown("""\n        <div class=\'property-card\'>\n            <h3 style=\'margin-top:0;\'>Property Summary</h3>\n        """, unsafe_allow_html=True)\n        \n        # Create two columns for details\n        detail_col1, detail_col2 = st.columns(2)\n        \n        with detail_col1:\n            st.markdown(f"""\n            - **Type**: {property_type}\n            - **City**: {city}\n            - **Area**: {area} sqm\n            - **Bedrooms**: {bedrooms}\n            """)\n        \n        with detail_col2:\n            st.markdown(f"""\n            - **Bathrooms**: {bathrooms}\n            - **Floor**: {level[0]}\n            - **Furnished**: {furnished}\n            - **Payment**: {payment}\n            """)\n        \n        st.markdown("</div>", unsafe_allow_html=True)\n        \n        # Disclaimer\n        st.markdown("""\n        <div style=\'margin-top:20px; font-size:14px; color:#7f8c8d;\'>\n        <i>Note: This estimate is based on current market trends and should be used as a reference only. \n        Actual prices may vary depending on specific property conditions and market fluctuations.</i>\n        </div>\n        """, unsafe_allow_html=True)\n\n# Footer\nst.markdown("---")\nst.markdown("""\n<div style=\'text-align:center; color:#7f8c8d; font-size:14px;\'>\n    <p>© 2023 Egypt Property Price Predictor | Powered by XGBoost AI Model</p>\n</div>\n""", unsafe_allow_html=True)')


# In[109]:


get_ipython().system('streamlit run app.py')


# # THANKS EPSILON :)
