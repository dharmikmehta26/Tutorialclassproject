
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


##reading the data

df=pd.read_excel("Enquiry 2018.xlsx")


# In[3]:


#df.drop(columns=['Name of Sudent','Month','Day','Year',' '],axis=1,inplace=True)
df = df.rename(columns={'%':'percentage','Handal':'Handle','School':'School_name'})
df.head()


# In[4]:


##converting date column to date
df['Date']=pd.to_datetime(df['Date'])
df.info()


# In[5]:


df['School_name'].unique()


# In[6]:


## Cleaning school data removing duplicate names,spelling errors.

df['School_name'].replace(['St. Xaviers','St Xaviers','St.Xaviars'],value='St. Xaviers',inplace=True)
df['School_name'].replace(['IDUBS','I.D.U.B.S'],value='I.D.U.B.S',inplace=True)
df['School_name'].replace(['Sahyadri','Sayadri'],value='Sahyadri',inplace=True)
df['School_name'].replace(['B.P.E.S','BPES'],value='B.P.E.S',inplace=True)
df['School_name'].replace(['nutan','Nutan'],value='Nutan',inplace=True)
df['School_name'].replace(['Shivaji','Shivai'],value='Shivaji',inplace=True)
df['School_name'].replace(['St. Agrsen','St. Agras'],value='St. Agrsen',inplace=True)


# In[7]:


#df['School_name'].nunique()
df['School_name']=df['School_name'].fillna('others')
#df['School_name'].isnull().sum()


# In[8]:


df['percentage']=df['percentage']*100
df['percentage'].describe()


# In[9]:


#replacing Nas with mean
df['percentage'].fillna(df['percentage'].mean(),inplace=True)
df['percentage'].isnull().sum()


# In[10]:


#rouding off upto 2 decimal values in percentage column
df['percentage']=df['percentage'].apply(lambda x:round(x,2))


# In[11]:


#df['Private class'].value_counts()
#df['Private class'].isnull().sum()


# In[12]:


df['Private class'].unique()

# private class column is not require so we will drop this columns
df.drop('Private class',axis=1,inplace=True)


# In[13]:


df['Std'].value_counts()


# In[14]:


##cleaning std column as it has spelling errors and multiple values with same name
df['Std'].replace(['Xl+Xll sci','Xll Sci','Xll sci','Xl+Xll Sci','XllSci','Xl+Xll+NEET','Xl+Xll','Only NEET'],value='Xll Sci',inplace=True)
df['Std'].replace(['Xl+Xll Com','Xl+Xll com','Xll com','Xll com','Xll Com','Xl+ Xll Com'],value='Xll Com',inplace=True)
df['Std'].replace(['IX+X','lX+X','X','X Privat','Vlll','only IT'],value='School',inplace=True)


# In[15]:



df['Std'].fillna('School',inplace=True)
df['Std'].value_counts()


# In[16]:


df['Handle'].value_counts()


# In[17]:


# Replacing null values with mode
df['Handle'].fillna(df['Handle'].mode()[0],inplace=True)
df['Handle'].isnull().sum()


# In[18]:


##labelencoding
from sklearn.preprocessing import LabelEncoder
label_en = LabelEncoder()
df['Handle_encoded']=label_en.fit_transform(df['Handle'])


# In[19]:


##  converting follow up column into binary columns
#df['Follow Up'].value_counts()
#df['Follow Up'].isnull().sum()
df['Follow Up'].replace('Done',value='1',inplace=True)
df['Follow Up'].fillna(0,inplace=True)


# In[20]:


df.head()


# In[21]:


# extracting fees from reason of not done column
d=df[df['Reason of not done'].astype(str).str.isdigit()]['Reason of not done']
a=d.index
df['fees']=df.loc[a]['Reason of not done']
df['fees'].fillna(0,inplace=True)


# In[22]:


# dropping unwanted columns
df.drop(columns=['Date','Name of Sudent','Contact','Ref','Area'],axis=1,inplace=True)


# In[23]:


df.columns


# In[24]:



df.head()


# In[25]:


# splittin std columns into stream and and std
df[['STD','Stream']] = df['Std'].astype(str).apply(lambda x: pd.Series(x.split(' ')))
#df.info()
df['Stream'].fillna('School',inplace=True)


# In[26]:



dummies = pd.get_dummies(df['Stream'])
df[['Commerce','School','Science']]=dummies


# In[27]:


#df.drop(['Std','STD'],axis=1,inplace=True)


# creating dummy variables for school_name
dumm = pd.get_dummies(df['School_name'])
df=pd.concat([df,dumm],axis=1)

#df['School_name_encoded']= label_en.fit_transform(df['School_name'])
df.drop('School_name',axis=1,inplace=True)


# In[28]:



# Creating admission status column based on student who have paid fees
df['Admission_status']= df['Reason of not done'].astype(str).str.isdigit()
df['Admission_status'].replace([False,True],value=[0,1],inplace=True)


# In[29]:


#df.drop('Reason of not done',inplace=True,axis=1)
print(df['Handle'].unique())
df['Handle_encoded'].unique()


# In[30]:



df.drop(['Reason of not done','Std','STD','Handle'],axis=1,inplace=True)
#df.to_csv('newdata.csv')
df.head()


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:



sns.countplot('Admission_status',data=df)


# In[33]:


dfcr=df.corr()
sns.heatmap(dfcr,cmap='coolwarm')


# In[34]:


g =sns.FacetGrid(data=df,col='Month',row='Stream')
g.map(plt.hist,'Admission_status')


# In[35]:


sns.countplot(x='Handle_encoded',data=df,hue='Admission_status')


# In[36]:


sns.countplot(x='Stream',data=df,hue='Admission_status')


# In[37]:


sns.distplot(df['percentage'])


# In[39]:


# month and day column was required for visual analysis,not required for model training.
df.drop(['Day','Month','Stream'],axis=1,inplace=True)


# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


x=df.drop('Admission_status',axis=1)
y=df['Admission_status']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# In[41]:


rfc=RandomForestClassifier(n_estimators=200)


# In[89]:


rfc.fit(x,y)


# In[90]:


from sklearn.externals import joblib
joblib.dump(rfc, 'model.pkl')


# In[43]:


rfc_pred= rfc.predict(X_test)


# In[44]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[45]:


print(confusion_matrix(y_test,rfc_pred))


# In[46]:


print(accuracy_score(y_test,rfc_pred))


# In[ ]:


from sklearn.svm import SVC
clf_3 = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

clf_3.fit(X_train,y_train)

# Predict on training set
pred_y_3 = clf_3.predict(X_test)



# How's our accuracy?
print( accuracy_score(y_test, pred_y_3) )


# In[ ]:


rfc.feature_importances_


# In[ ]:


df.columns


# In[47]:



len(df.columns)


# In[48]:


df.columns


# In[86]:


#['Vidya' 'Hitesh' 'Shwetambari' 'Geeta mam' 'Nilakshi' 'Pradipa']
#array([5, 1, 4, 0, 2, 3]
def if_el(x):
    if x == 'Vidhya':
        return 5
    elif x=='Hitesh':
        return 1
    elif x=='Shwetambari':
        return 4
    elif x=='Geeta mam':
        return 0
    elif x=='Nilakshi':
        return 2
    else:
        return 3


# In[87]:


abc = if_el('Nilakshi')


# In[88]:



print(abc)


# In[70]:


z = 'Y.C.S'


# In[83]:


cols=df.columns
cols[77]


# In[82]:


cols.get_loc(z)
