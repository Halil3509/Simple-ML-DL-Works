#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


#Standardizasyon için sklearn'ün StandardScaler adlı fonksiyonu vardır fakat ben elle gerçekleştireceğim.
def standardization(array):
    """
    :param array = must be numpy array format
    """
    
    try:
        if not isinstance(array,np.ndarray):
             raise TypeError("array varibale must be numpy array")
        print("Standardize etme işlemi başlamıştır")
        mean = array.mean()
        std = array.std()
        print("Standardize edilmeden önceki ortalama:", mean)
        print("Standardize edilmeden önceki standard sapma:", std)
        new_list = []
        print("Standardize etme işlemi başarıyla gerçekleşmiştir")
        
        for value in array:
            new_list.append(((value - mean)/ std).round(2))
        return np.array(new_list)
    except Exception as err:
        raise Exception("Fault:", err)


def three_sigma(data):
    """
    compute a three sigma low and upper edges
    
    :param data => data that will perform three sigma
    
    return => According to three_sigma low_edge, upper_egde for variable
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be numpy array format. Please you check it")
    
    print("Three Sigma process started.")
    low_edge = -3*data.std()  + data.mean()
    upper_edge = 3*data.std() + data.mean()
    print("Three Sigma precess completed successfully")
    print("Low Edge:", low_edge)
    print("Upper Edge:", upper_edge)

    return low_edge, upper_edge

# In[8]:


def min_max_scaling(array):
    """
    doing normalization to array
    
    :param array => must be numpy array format
    
    formula = x - x_min / x_max - x_min
    """
    try:
        print("Min Max Scaling started")
        new_list = []
        for i in array:
            new_list.append((i-array.min())/(array.max()- array.min()))
        print("Min Max Scaling finished Succesfully")
        return np.array(new_list)
    except TypeError():
        raise TypeError("Array must be numpy array")
    


# In[9]:


def robust_scaler(series):
    """
    We want to series type Pandas.Series. Because we get a quantile of data. That is a only different between other scales and 
    robust scaler
    
    :param Series => must be pandas.Series
    """
    try:
        print("Robust Scaling started")
        new_list = []
        Q3 = series.quantile(0.75)
        Q1 = series.quantile(0.25)
        median = series.median()
        for x in series:
            new_list.append((x - median)/(Q3-Q1))
        print("Robust Scaling finished succesfully")
        return np.array(new_list) 
    except TypeError():
        raise TypeError("Series must be Pandas.Series format")
    


# In[10]:


def anomaly_detection_with_IQR(series , coff):
    """
    :param series => must be series format
    :param coff => cofficent of IQR for finding low and upper edge

    """
    if not isinstance(series, pd.Series):
        raise TypeError("Series must be Pandas Series format")

    try:
            print("Anomaly Detection precess with finding IQR started")
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
    
            IQR = Q3 -Q1
    
            low_edge = Q1 - coff*IQR
            upper_edge = Q3 + coff*IQR
            
            print("Cofficent:", coff)
            print("Low Edge:",low_edge)
            print("Upper Edge:",upper_edge)
            
            
            print("Anomaly detection process with finding IQR finished succesfully.")
            return low_edge, upper_edge
        
    except TypeError():
        raise TypeError("Array that came must be numy array.")


# In[11]:


def IQR_edges_plot(standardized_data,low_edge, upper_edge, title):
    """
    Draw a plot with low_edge and upper_edge
    
    :param standardized_data => data
    :param low_edge => low edge that came IQR calculation
    :param upper_edge => upper edge that came IQR calculation
    :param title => title of plot
    """
    plt.figure(figsize =(12,10))
    plt.title(title)
    plt.axvline(x = low_edge, c = "b", linestyle ='--')
    plt.axvline(x = upper_edge, c = "b", linestyle ='--')
    sns.kdeplot(standardized_data, fill = True,  color = 'g')
    plt.show();    


# In[ ]:





# In[ ]:




