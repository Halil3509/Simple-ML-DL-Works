import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


def get_datasets(general_folder_name, folder_name, sep):
    """
    """
    if not isinstance(general_folder_name, str) or not isinstance(folder_name, str):
        raise TypeError("general_folder_name and folder_name must be string format")

    # Getting UI_output.json
    f = open('../../../mvp/data/{0}/{1}/UI_Output.json'.format(general_folder_name, folder_name))
    ui_output_json = json.load(f)

    # Getting Datasets
    real_df = pd.read_csv('../../../mvp/data/{0}/{1}/Preprocessed.csv'.format(general_folder_name, folder_name),
                          sep=sep)
    product_df = pd.read_csv('../../../mvp/data/{0}/{1}/SyntheticData.csv'.format(general_folder_name, folder_name),
                             sep=sep)

    return real_df, product_df, ui_output_json


def draw_shape_comparison(real, product):
    """
    """
    plt.subplots(1,2,figsize =(20,10))
    plt.suptitle("Shapes Comparison", fontsize = (25))

    plt.subplot(121)
    plt.title("X Shapes Comparison",fontsize = 20)
    plt.bar(1, real.shape[0], color = 'g', width = 0.30, edgecolor = 'black',label = "Real Data")
    plt.text(0.95, real.shape[0]+20, real.shape[0])
    plt.bar(2, product.shape[0], color = 'r', width = 0.30, edgecolor = 'black',label = "Product Data")
    plt.text(1.95, product.shape[0]+20, product.shape[0])
    plt.xlabel('Real', fontweight ='bold', fontsize = 15)
    plt.ylabel('Product', fontweight ='bold', fontsize = 15)
    plt.xticks([1,2],['Real', 'Product'], fontsize = 12)
    plt.legend()

    plt.subplot(122)
    plt.title("Y Shapes Comparison", fontsize = 20)
    plt.bar(1, real.shape[1], color = 'g', width = 0.30, edgecolor = 'black',label = "Real Data")
    plt.text(0.95, real.shape[1], real.shape[1])
    plt.bar(2, product.shape[1], color = 'r', width = 0.30, edgecolor = 'black',label = "Product Data")
    plt.text(1.95, product.shape[1], product.shape[1])
    plt.xlabel('Real', fontweight ='bold', fontsize = 15)
    plt.ylabel('Product', fontweight ='bold', fontsize = 15)
    plt.xticks([1,2],['Real', 'Product'], fontsize = 12)
    plt.legend()

    plt.show()


def draw_heatmaps(real, product):
    """
    """
    plt.subplots(3, 1, figsize=(15, 36))
    plt.suptitle("Correlations of Datas", fontsize=27)

    plt.subplot(311)
    plt.title("Real Data", fontsize=20)
    sns.heatmap(real.corr(), cmap='YlGnBu')

    plt.subplot(312)
    plt.title("Product Data", fontsize=20)
    sns.heatmap(product.corr(), cmap='YlGnBu')

    plt.subplot(313)
    plt.title("Difference of Datas", fontsize=20)
    diff = round(real.corr() - product.corr(), 2)
    sns.heatmap(diff, cmap='YlGnBu')
    plt.subplots_adjust(top=0.95, hspace=0.4)

    plt.show()


def draw_num_dist(real, product, num_cols):
    plt.subplots(len(num_cols), 1, figsize = (15, len(num_cols)*10))
    plt.suptitle("Distributions of Numeric Features", fontsize = 30)

    for index, col in enumerate(num_cols):
        plt.subplot(len(num_cols), 1,index+1)
        plt.title("%s Distribution Plot" %num_cols[index])
        sns.distplot(real[col], color ='g', label = 'Real', kde_kws = {'linewidth':3})
        sns.distplot(product[col], color = 'r', label= 'Product',kde_kws = {'linewidth':3})
        plt.legend()
    plt.subplots_adjust(top=0.97)
    plt.show()


def draw_dist_categoric(real, product, categoric_cols):
    """
    """
    plt.subplots(len(categoric_cols), 1, figsize = (15, len(categoric_cols)*10))
    plt.suptitle("Distributions of Categoric Features", fontsize = 20)

    for index,col in enumerate(categoric_cols):
        plt.subplot(len(categoric_cols),1,index +1)
        plt.title("%s Distribution Plot" %col, fontsize = 15)
        sns.countplot(real[col], color ='g',label = 'Real', alpha = 0.5)
        sns.countplot(product[col], color ='r',label = 'Product', alpha = 0.5)
        plt.legend()
    plt.subplots_adjust(top=0.97)
    plt.show()


def apply_pca(real, product, dimension):
    """
    """
    sc = StandardScaler()
    real = sc.fit_transform(real)
    product = sc.fit_transform(product)

    if dimension == 2:
        pca2 = PCA(n_components=2)
        real_reduced = pca2.fit_transform(real)
        product_reduced = pca2.fit_transform(product)
        print("2-component PCA applied")
    elif dimension == 3:
        pca3 = PCA(n_components=3)
        real_reduced = pca3.fit_transform(real)
        product_reduced = pca3.fit_transform(product)
        print("3-component PCA applied")
    else:
        raise ValueError("Dimension must be 2 or 3. Please check dimension parameter")

    return real_reduced, product_reduced


def pca_visualization(real_reduced, product_reduced, dimension, hue_real, hue_product):
    """
    """

    if real_reduced.shape[1] != dimension:
        raise ValueError("Please Enter dimension parameter value correctly.")

    if dimension == 3:
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Comparison Real And Product Datas in 3-Dimension", fontsize=25)

        # For Real Data
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title("Real Data", fontweight='bold', fontsize=15)
        ax.set_xlabel("1. Component")
        ax.set_ylabel("2. Component")
        ax.set_zlabel("3. Component")
        ax.scatter3D(real_reduced[:, 0], real_reduced[:, 1], real_reduced[:, 2], s=10, lw=1, c=hue_real,cmap = 'BuGn')

        # For Product Data
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title("Product Data", fontweight='bold', fontsize=15)
        ax2.set_xlabel("1. Component")
        ax2.set_ylabel("2. Component")
        ax2.set_zlabel("3. Component")
        ax2.scatter3D(product_reduced[:, 0], product_reduced[:, 1], product_reduced[:, 2],
                      s=10, lw=1, c=hue_product, cmap = 'BuGn')
        plt.legend()
        plt.show()



    elif dimension == 2:
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle("Comparison Real And Product Datas in 2-Dimension", fontsize=25)

        # For Real Data
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title("Real Data", fontweight='bold', fontsize=15)
        ax.set_xlabel("1. Component")
        ax.set_ylabel("2. Component")
        ax.scatter(real_reduced[:, 0], real_reduced[:, 1], s=10, lw=1, c=hue_real,cmap = 'BuGn')

        # For Product Data 
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("Product Data", fontweight='bold', fontsize=15)
        ax2.set_xlabel("1. Component")
        ax2.set_ylabel("2. Component")
        ax2.scatter(product_reduced[:, 0], product_reduced[:, 1],
                    s=10, lw=1, c=hue_product,cmap = 'BuGn')
        plt.show()


def draw_dist_comparison_plot(real, product, variable_1, variable_2):
    plt.subplots(1,2, figsize= (20,10))
    plt.suptitle("weight and insurance_cost distributions", fontsize = 20)

    plt.subplot(121)
    sns.regplot(data = real, x = variable_1, y = variable_2,
                line_kws = {'color':'g', 'linewidth':6}, scatter_kws = {'alpha':0.2, 'color':'g'})
    plt.title("Real Data", fontsize = 15)

    plt.subplot(122)
    sns.regplot(data = product, x = variable_1, y = variable_2,
                line_kws = {'color':'r', 'linewidth':6}, scatter_kws = {'alpha':0.2, 'color':'r'})
    plt.title("Product Data", fontsize = 15)

    plt.show()


def draw_kde_plot_comparison(real, product, variable):
    """
    """

    plt.subplots(1, 2, figsize=(20, 10))
    plt.suptitle("Distribution of Features in Kde Plot")

    plt.subplot(121)
    plt.title("Real Data", fontsize=15)
    sns.kdeplot(x=variable, data=real, fill=True, color='g')

    plt.subplot(122)
    plt.title("Product Data", fontsize=15)
    sns.kdeplot(x=variable, data=product, fill=True, color='r')

    plt.show()


def finding_number_iqr_outlier(df, variable):
    """
    """
    q_1 = df[variable].quantile(0.25)
    q_3 = df[variable].quantile(0.75)
    iqr = q_3 - q_1

    lower_bound = q_1 - 1.5 * iqr
    upper_bound = q_3 + 1.5 * iqr

    number_outlier = len(df[(df[variable] < lower_bound) | (df[variable] > upper_bound)])
    return number_outlier


def draw_comparison_boxplot(real, product, variable):
    """
    """
    plt.subplots(2, 1, figsize=(20, 10))
    plt.suptitle("Distribution of %s Feature in Boxplot" % variable, fontsize=25)

    plt.subplot(211)
    plt.title("Real Data", fontsize=15)
    sns.boxplot(x=real[variable], color='g')
    sns.stripplot(x=real[variable], color='g', alpha=0.5)

    plt.subplot(212)
    plt.title("Product Data", fontsize=15)
    sns.boxplot(x=real[variable], color='r')
    sns.stripplot(x=real[variable], color='r', alpha=0.5)

    plt.subplots_adjust(hspace=0.4)
    plt.show()


def split_dataset(data, target):
    """
    """
    features = data.drop(target, axis = 1)
    label = data[target]
    return train_test_split(features, label, test_size= 0.33, random_state=42)


def model_learning(model_variable, data,  split_dataset, target):
    """

    """
    x_train, x_test, y_train, y_test = split_dataset(data, target)
    pred = model_variable.fit(x_train, y_train).predict(x_test)
    return pred, y_test


def model_evaluating(real_pred, product_pred, real, product,apply_pca ,pca_visalization, dimension, target, real_actual,
                    product_actual):
    real_reduced, product_reduced = apply_pca(real, product, dimension)
    pca_visalization(real_reduced, product_reduced, dimension, hue_real = real[target], hue_product = product[target])
    print("Real Data Classification:\n", classification_report(real_pred, real_actual))
    print("Product Data Classification:\n", classification_report(product_pred, product_actual))



def apply_if(num_cols_df):
    """
    """
    isolation_forest = IsolationForest(contamination=0.009)
    if_anomalies = isolation_forest.fit(num_cols_df).predict(num_cols_df)
    return if_anomalies


