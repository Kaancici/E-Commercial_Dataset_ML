# E-Commercial Dataset Description
In this project, we performed Exploratory Data Analysis (EDA) and applied various machine learning techniques to analyze whether products would arrive on time. Our goal was to thoroughly analyze the data, identify the best-performing model, and accurately predict delivery timelines to determine if the products would be delivered on time.

#### 1. Numeric Variables (int64):                                         
ID: Unique identifier for each observation.                                   
Customer_care_calls: Number of calls made to customer service.                         
Customer_rating: Customer satisfaction rating (likely on a scale from 1 to 5).                 
Cost_of_the_Product: Cost of the product.                                           
Prior_purchases: Number of previous purchases.                             
Discount_offered: Amount of discount offered.                           
Weight_in_gms: Weight of the product in grams.                         
Reached on time: It is the target variable, where 1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time.                      

#### 3. Categorical Variables (object):                                                      
Warehouse_block: Warehouse blocks (e.g., A, B, C, D, F).                    
Mode_of_Shipment: Mode of shipment (e.g., 'Ship', 'Flight', 'Road').                   
Product_importance: Importance of the product (e.g., 'low', 'medium', 'high').                 
Gender: Customer's gender (e.g., 'Male', 'Female').                      


## Libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,cohen_kappa_score
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
```

## Load and Checked the Data 
```
train_df = pd.read_csv(r'C:\...\train.csv')
```
```
train_df.head()
```
```
Then checked the columns
```
```
train_df.columns
```
```
The data was described
```
```
train_df.describe()
```

## Visualizing the E-Commercial Dataset
We made a function for bar plot visualizing the data.
### Categorical Visualization
```
def bar_plot(variable):

    var = train[variable]
    varValue = var.value_counts()
    
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))

category1 = ["Warehouse_block","Mode_of_Shipment","Product_importance","Gender"]
for c in category1:
    bar_plot(c)
```

### Numeric Visualization
```
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
num1 = ["ID","Customer_care_calls","Customer_rating","Cost_of_the_Product","Prior_purchases","Discount_offered","Weight_in_gms","Reached.on.Time_Y.N"]
for n in num1:
    plot_hist(n)
```
![image](https://github.com/user-attachments/assets/9816ff34-73bd-47df-af65-157bf56b9de6)
![image](https://github.com/user-attachments/assets/89363a81-b7d7-482c-9c4b-108628182e52)
![image](https://github.com/user-attachments/assets/bf0ea670-1dd0-4f8c-983e-940468050799)
![image](https://github.com/user-attachments/assets/5217084a-7f64-45bf-92e8-83936f7cda4d)
![image](https://github.com/user-attachments/assets/641f9d92-347f-4d34-94af-7fe603a6a592)
![image](https://github.com/user-attachments/assets/e999aca0-81ca-4e89-aea2-d6c9265972b1)
![image](https://github.com/user-attachments/assets/ea7bbcb4-f085-48d8-8820-9b561bb44c75)
![image](https://github.com/user-attachments/assets/0be6e193-f249-46ca-af51-1a659278254c)

1. Customer Rating Distribution:
Comment: The distribution of customer ratings (ranging from 1 to 5) appears to be evenly distributed. This indicates that customers provide a balanced set of ratings, with no particular dominance of extremely high or low ratings. The differences between the ratings are minimal, and each rating is used approximately the same number of times.
2. Prior Purchases Distribution:
Comment: This graph shows that the number of prior purchases is mostly concentrated between 2 and 4. However, the number of purchases above 5 significantly decreases. This suggests that customers generally make a limited number of repeat purchases, and the number of customers who frequently shop is relatively low.
3. Discount Offered Distribution:
Comment: The majority of discounts are concentrated in the 0-10 range, indicating that the company generally prefers to offer low discount rates. Higher discount rates are quite rare, suggesting that this strategy is rarely used. Additionally, there is a noticeable decline in the frequency of higher discounts, implying that low discounts are more common and higher discounts are less frequently offered.
4. Reached on Time Distribution:
Comment: Most products were delivered on time (represented by 1), with a smaller portion being delivered late (represented by 0). The number of products not delivered on time is relatively low, indicating that the overall delivery performance is good. However, some delays in delivery are also observed.
5. Customer Care Calls Distribution:
Comment: The distribution of customer care calls shows that most customers made a low number of calls. This could indicate that overall customer satisfaction is high, and major issues are rarely encountered. If there is a noticeable increase in call frequency on the graph, it might suggest a rise in customer complaints during a specific period.
6. Cost of the Product Distribution:
Comment: Assuming that the product costs are distributed over a wide range, it suggests that there is a variety in pricing strategies and a broad product range. The average cost levels might be concentrated within a certain range, providing insight into the company's pricing strategies.
7. Weight in Grams Distribution:
Comment: The distribution of product weights might show a concentration in a specific weight range. This indicates that products are generally grouped into specific weight categories, and the company might be focusing on certain types of products.

## Basic EDA
Here, we examined the probabilities of whether products arrived on time by relating them to other parameters.
```
train[["Reached.on.Time_Y.N","Mode_of_Shipment"]].groupby("Mode_of_Shipment", as_index=False).mean().sort_values(by="Reached.on.Time_Y.N",ascending = False)
```
![image](https://github.com/user-attachments/assets/d0cd6c73-66e7-496b-89e7-6f06fb7de24e)
The "Flight" method has the highest probability of on-time delivery, while the "Road" method has the lowest probability.

```
train[["Reached.on.Time_Y.N","Product_importance"]].groupby("Product_importance", as_index=False).mean().sort_values(by="Reached.on.Time_Y.N",ascending = False)
```
![image](https://github.com/user-attachments/assets/269d562a-e86f-4898-a96e-cb06c8f2f687)
Products with "high" importance have the highest likelihood of being delivered on time, while those with "medium" importance have the lowest likelihood of on-time delivery.

```
train[["Reached.on.Time_Y.N","Prior_purchases"]].groupby("Prior_purchases", as_index=False).mean().sort_values(by="Reached.on.Time_Y.N",ascending = False)
```
![image](https://github.com/user-attachments/assets/9672f184-3c53-484b-b6ba-0043d82ca5ce)
Customers who have made 7 prior purchases have the highest probability (approximately 67.6%) of receiving their products on time. This suggests that customers with a moderate history of prior purchases tend to experience better delivery performance. Conversely, customers with 5 prior purchases have the lowest probability (approximately 49.9%) of on-time delivery, indicating that as the number of prior purchases reaches this level, the likelihood of timely delivery decreases.
Interestingly, the relationship between prior purchases and on-time delivery is not strictly linear. For example, customers with 3 prior purchases have a relatively high on-time delivery rate (around 64.1%), while those with 4 or 5 prior purchases see a noticeable decline in this probability. This suggests that other factors, possibly related to order complexity or customer history, might influence the delivery performance in these cases.

```
train[["Reached.on.Time_Y.N","Customer_rating"]].groupby("Customer_rating", as_index=False).mean().sort_values(by="Reached.on.Time_Y.N",ascending = False)
```
![image](https://github.com/user-attachments/assets/12c319c0-8763-4135-93fc-9fd348b6c429)
Customers who gave a rating of 5 have the highest probability of receiving their products on time (approximately 60.7%). This might indicate that higher satisfaction is associated with better delivery performance, possibly due to better service or less complex orders.
Interestingly, customers who gave a rating of 3 also have a similar on-time delivery probability (60.6%), suggesting that moderate satisfaction levels do not significantly differ from the highest satisfaction levels in terms of delivery performance.
On the other hand, customers who gave lower ratings (1 and 2) have a slightly lower probability of on-time delivery (around 58.7-59.5%). This could imply that lower satisfaction might be related to delays or issues in the delivery process, contributing to the lower ratings.
Overall, the data suggests a slight positive correlation between higher customer ratings and the likelihood of on-time delivery, though the difference between the highest and lowest ratings is not very large.

```
train[["Reached.on.Time_Y.N","Warehouse_block"]].groupby("Warehouse_block", as_index=False).mean().sort_values(by="Reached.on.Time_Y.N",ascending = False)
```
![image](https://github.com/user-attachments/assets/6a492471-15aa-4aca-9c18-3a183cbf09b9)
The differences in on-time delivery rates among the blocks are relatively small, but Block B consistently performs slightly better than the others.
Overall, while there are some variations in performance among the warehouse blocks, they all maintain a relatively close range of on-time delivery probabilities.

```
train[["Reached.on.Time_Y.N","Customer_care_calls"]].groupby("Customer_care_calls", as_index=False).mean().sort_values(by="Reached.on.Time_Y.N",ascending = False)
```
![image](https://github.com/user-attachments/assets/b1f4e7ef-f77b-477f-9343-450f9d8798f7)
Customers who made fewer calls (2-3 calls) have the highest probability of on-time delivery, with the highest being 65.2% for those who made 2 calls.
As the number of customer care calls increases, the probability of on-time delivery decreases, with the lowest probability (approximately 51.6%) observed for those who made 6 or 7 calls.
This pattern suggests that customers who needed to make more calls to customer care might have experienced more issues with their orders, possibly leading to delays in delivery.

## Outlier Detection and Removal
We created a function to detect and remove outliers:
```
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        Q1 = np.percentile(df[c],25)
        Q3 = np.percentile(df[c],75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers

liste = ["Customer_care_calls","Customer_rating","Cost_of_the_Product","Discount_offered"]
train.loc[detect_outliers(train,liste)]
```
And data has no outlier, so we can continue.

## Find Missing Value
We need to check missing values and fill them.
```
train_len = len(train)
train_len
```
```
train.isnull().sum()
```
As you can see we checked but didn't find any missing values.

## Correlation
Corellation shows us the reltions between our variables.
Since ID is my index variable, I don't need to consider its correlation.
```
list = ["ID","Customer_care_calls","Customer_rating","Cost_of_the_Product","Prior_purchases","Discount_offered","Weight_in_gms","Reached.on.Time_Y.N"]
sns.heatmap(train[list].corr(),annot=True,fmt=".2f")
```
![image](https://github.com/user-attachments/assets/8fe5ac01-9698-4f89-8c77-3122960f15a4)
Discount_offered and Reached.on.Time_Y.N (0.40):

Interpretation: There is a moderate positive correlation between the discount offered and the likelihood of on-time delivery. This suggests that when higher discounts are offered, the products are more likely to be delivered on time. This might be due to promotional campaigns where faster delivery is prioritized to enhance customer satisfaction.

Customer_care_calls and Cost_of_the_Product (0.32):
Interpretation: A moderate positive correlation exists between the number of customer care calls and the cost of the product. Higher-priced products tend to generate more customer service interactions, likely because customers are more concerned or have more questions when they spend more money.

Discount_offered and Weight_in_gms (-0.38):
Interpretation: There is a moderate negative correlation between the discount offered and the weight of the product. Heavier products tend to receive lower discounts, possibly due to higher shipping costs or handling fees associated with heavier items.

Reached.on.Time_Y.N and Weight_in_gms (-0.27):
Interpretation: A slight negative correlation between product weight and on-time delivery suggests that heavier products are less likely to be delivered on time. This might be due to logistical challenges associated with handling and shipping heavier items.

Cost_of_the_Product and Customer_care_calls (0.32):
Interpretation: There is a moderate positive correlation between the product's cost and the number of customer care calls. This might indicate that customers who purchase more expensive products are more likely to contact customer service, perhaps to ensure that the product is delivered safely or to address any concerns they have about the purchase.

Reached.on.Time_Y.N and Customer_care_calls (-0.07):
Interpretation: There is a very weak negative correlation between the number of customer care calls and the likelihood of on-time delivery. This suggests that more customer service calls could be slightly related to delivery issues, although the correlation is weak.

## Data Cleaning
In this step, I reviewed the correlation table and identified the variables that had either a high or low correlation with on-time delivery. Based on this analysis, I determined that these specific variables should be removed from my dataset.
```
train.drop(["ID","Customer_care_calls","Gender","Cost_of_the_Product","Prior_purchases","Customer_rating"],axis=1,inplace=True)
```
## Chancing Data to numeric
```
warehouse_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
shipment_mapping = {'Ship': 1, 'Flight': 2, 'Road': 3}
importance_mapping = {'low': 1, 'medium': 2, 'high': 3}

train['Warehouse_block'] = train['Warehouse_block'].map(warehouse_mapping)
train['Mode_of_Shipment'] = train['Mode_of_Shipment'].map(shipment_mapping)
train['Product_importance'] = train['Product_importance'].map(importance_mapping)

print(df)
```
## Modeling
We split our data
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
We previously identified class imbalance, with a significant difference between on-time and late deliveries. To address this, we used SMOTE to balance the classes, ensuring that our model can learn effectively and make accurate predictions.
```
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print("SMOTE sonrası sınıf dağılımı:\n", pd.Series(y_resampled).value_counts())
```
### Building Models and Selecting the Best Model"
This function, evaluate_model, is designed to evaluate a machine learning model's performance on both training and testing datasets using several key metrics.
```
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'Set': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Kappa': []
    }
    
    for (X, y, y_pred, dataset) in [(X_train, y_train, y_train_pred, 'Train'), (X_test, y_test, y_test_pred, 'Test')]:
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        kappa = cohen_kappa_score(y, y_pred)

        metrics['Set'].append(dataset)
        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1 Score'].append(f1)
        metrics['Kappa'].append(kappa)
        
        print(f"Evaluating on {dataset} data")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Kappa: {kappa}")
        print(f"Confusion Matrix:\n {confusion_matrix(y, y_pred)}")
        print("\n")
    
    return metrics

```
The code is setting up multiple machine learning models along with a comprehensive set of hyperparameters for each. This setup is typically used in model selection processes, where the best model and hyperparameter combination are chosen based on performance on the validation set.
```
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier(),
             XGBClassifier(),
             LGBMClassifier()]


dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C": np.logspace(-3, 3, 7),
                     "penalty": ["l1"],
                     "solver": ["liblinear", "saga"]}

knn_param_grid = {"n_neighbors": [3, 5, 7, 9, 11],
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan"]}
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 50, 100],
    'boosting_type': ['gbdt', 'dart'],
    'objective': ['binary'],
    'colsample_bytree': [0.8, 1.0],
    'subsample': [0.8, 1.0]
}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid,xgb_param_grid,
                   lgbm_param_grid]
```
Then we run the function.
```
best_estimators = []
for clf, param in zip(classifier, classifier_param):
    grid_search = GridSearchCV(estimator=clf, param_grid=param, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)
    best_estimators.append(grid_search.best_estimator_)

    print(f"Best parameters for {clf.__class__.__name__}: {grid_search.best_params_}")
    print(f"Best score for {clf.__class__.__name__}: {grid_search.best_score_}")
```
![image](https://github.com/user-attachments/assets/98340be7-caa7-46c7-8d99-814c85a20ad8)
The best scores were obtained with different hyperparameter methods for each model. SVC and XGBClassifier provided the highest accuracy rates, while Logistic Regression and RandomForestClassifier showed lower performance than other models. These results show that each model can be optimized for different robustness for specific data books and problems.

### Checkin Performance and Overfits
```
for model in best_estimators:
    print(f"Evaluating {model.__class__.__name__}")
    evaluate_model(model, X_resampled, y_resampled, X_test, y_test)
```
Evaluating DecisionTreeClassifier
Evaluating on Train data
Accuracy: 0.7249180327868853
Precision: 0.8061867389355413
Recall: 0.7249180327868853
F1 Score: 0.7053675467429364
Kappa: 0.44983606557377054
Confusion Matrix:
 [[4495   80]
 [2437 2138]]


Evaluating on Test data
Accuracy: 0.6706060606060606
Precision: 0.7987020468295963
Recall: 0.6706060606060606
F1 Score: 0.6600031453081954
Kappa: 0.3924292734219015
Confusion Matrix:
 [[1277   35]
 [1052  936]]


Evaluating SVC
Evaluating on Train data
Accuracy: 0.9926775956284153
Precision: 0.9927388271343726
Recall: 0.9926775956284153
F1 Score: 0.9926773681368153
Kappa: 0.9853551912568306
Confusion Matrix:
 [[4567    8]
 [  59 4516]]


Evaluating on Test data
Accuracy: 0.6393939393939394
Precision: 0.6375036819457129
Recall: 0.6393939393939394
F1 Score: 0.6383457480092722
Kappa: 0.24307841761562965
Confusion Matrix:
 [[ 696  616]
 [ 574 1414]]


Evaluating RandomForestClassifier
Evaluating on Train data
Accuracy: 0.8220765027322404
Precision: 0.8424538798727104
Recall: 0.8220765027322404
F1 Score: 0.8193897443754168
Kappa: 0.6441530054644808
Confusion Matrix:
 [[4319  256]
 [1372 3203]]


Evaluating on Test data
Accuracy: 0.6578787878787878
Precision: 0.704083596424011
Recall: 0.6578787878787878
F1 Score: 0.6594897622348603
Kappa: 0.33643544168888817
Confusion Matrix:
 [[1042  270]
 [ 859 1129]]


Evaluating LogisticRegression
Evaluating on Train data
Accuracy: 0.6856830601092896
Precision: 0.7064179253112033
Recall: 0.6856830601092896
F1 Score: 0.6775863758882626
Kappa: 0.3713661202185793
Confusion Matrix:
 [[3862  713]
 [2163 2412]]


Evaluating on Test data
Accuracy: 0.6448484848484849
Precision: 0.705800105010501
Recall: 0.6448484848484849
F1 Score: 0.6441095436728158
Kappa: 0.3208922726825775
Confusion Matrix:
 [[1080  232]
 [ 940 1048]]


Evaluating KNeighborsClassifier
Evaluating on Train data
Accuracy: 0.9979234972677595
Precision: 0.9979303724507814
Recall: 0.9979234972677595
F1 Score: 0.9979234900998972
Kappa: 0.9958469945355192
Confusion Matrix:
 [[4574    1]
 [  18 4557]]


Evaluating on Test data
Accuracy: 0.6409090909090909
Precision: 0.6582792128422358
Recall: 0.6409090909090909
F1 Score: 0.6446543518180189
Kappa: 0.2776029385793467
Confusion Matrix:
 [[ 865  447]
 [ 738 1250]]


Evaluating XGBClassifier
Evaluating on Train data
Accuracy: 0.7227322404371584
Precision: 0.8165876248782388
Recall: 0.7227322404371584
F1 Score: 0.7005376301293004
Kappa: 0.4454644808743169
Confusion Matrix:
 [[4552   23]
 [2514 2061]]


Evaluating on Test data
Accuracy: 0.6666666666666666
Precision: 0.8090608759596758
Recall: 0.6666666666666666
F1 Score: 0.6535758649809784
Kappa: 0.3888716044494188
Confusion Matrix:
 [[1297   15]
 [1085  903]]


Evaluating LGBMClassifier
Evaluating on Train data
Accuracy: 0.7323497267759563
Precision: 0.8003604529248308
Recall: 0.7323497267759563
F1 Score: 0.7162895624942686
Kappa: 0.46469945355191256
Confusion Matrix:
 [[4439  136]
 [2313 2262]]


Evaluating on Test data
Accuracy: 0.6696969696969697
Precision: 0.7758105790525577
Recall: 0.6696969696969697
F1 Score: 0.662416455797523
Kappa: 0.38476474306872566
Confusion Matrix:
 [[1233   79]
 [1011  977]]

 As a result, the SVC and KNeighborsClassifier models are likely experiencing overfitting, while models such as DecisionTree, RandomForest, XGBoost, and LGBM demonstrate more balanced performance. However, the performance of each model on the test data generally indicates that the model’s ability to generalize to real-world data needs to be further optimized.
 
```
def plot_model_performance(models, X_train, y_train, X_test, y_test):
    performance_metrics = []

    for model in models:
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        for i in range(len(metrics['Set'])):
            performance_metrics.append({
                'Model': model.__class__.__name__,
                'Set': metrics['Set'][i],
                'Accuracy': metrics['Accuracy'][i],
                'Precision': metrics['Precision'][i],
                'Recall': metrics['Recall'][i],
                'F1 Score': metrics['F1 Score'][i],
                'Kappa': metrics['Kappa'][i]
            })

    performance_df = pd.DataFrame(performance_metrics)
    performance_melted = performance_df.melt(id_vars=['Model', 'Set'], var_name='Metric', value_name='Value')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Metric', y='Value', hue='Model', data=performance_melted, ci=None)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
plot_model_performance(best_estimators, X_resampled, y_resampled, X_test, y_test)
```
![image](https://github.com/user-attachments/assets/8564b38c-b487-4149-883f-4ae64ea0fe71)

 While the SVC and KNeighborsClassifier models are potentially experiencing overfitting, models like RandomForest, XGBoost, and LGBM exhibit a more balanced and generalizable performance.

### Combining Different Models
This code creates a soft voting ensemble classifier that combines the best-performing DecisionTree, RandomForest, and KNeighbors models. The combined model is trained on the resampled dataset, and then its performance, along with the individual models, is evaluated and visualized.
```
votingC = VotingClassifier(estimators=[("dt", best_estimators[0]),
                                       ("rfc", best_estimators[2]),
                                       ("knn", best_estimators[4])],
                           voting="soft", n_jobs=-1)

votingC = votingC.fit(X_resampled, y_resampled)
all_models = best_estimators + [votingC]
plot_model_performance(all_models, X_resampled, y_resampled, X_test, y_test)
```

Evaluating on Train data
Accuracy: 0.9960655737704918
Precision: 0.9960962914769032
Recall: 0.9960655737704918
F1 Score: 0.9960655128657729
Kappa: 0.9921311475409836
Confusion Matrix:
 [[4575    0]
 [  36 4539]]


Evaluating on Test data
Accuracy: 0.6624242424242425
Precision: 0.7143910094057587
Recall: 0.6624242424242425
F1 Score: 0.6632864073294695
Kappa: 0.3486238922247359
Confusion Matrix:
 [[1071  241]
 [ 873 1115]]

This evaluation suggests that the model performs exceptionally well on the training data, with very high accuracy, precision, recall, F1 score, and a Kappa score close to 1, which indicates near-perfect agreement. However, the performance drops significantly on the test data, with a lower accuracy of around 66%, and a moderate Kappa score of 0.35, indicating less agreement and potential overfitting. The confusion matrix shows that while the model is good at identifying true positives (1115 correctly classified as positive), it struggles with false positives (241 incorrectly classified as positive) and false negatives (873 incorrectly classified as negative), reflecting the model's difficulty in generalizing to unseen data.

![image](https://github.com/user-attachments/assets/ac4631e6-0216-4d5c-88cf-227644cfe79a)

The VotingClassifier, which combines several models, tends to outperform or match the best individual models across most metrics, suggesting that an ensemble approach may provide the most balanced and reliable predictions in this scenario. However, models like KNeighborsClassifier show signs of overfitting, with high variance between training and test performance.

```
votingk = VotingClassifier(estimators=[
    ("dt", best_estimators[0]),
    ("logreg", best_estimators[3]),
    ("knn", best_estimators[4]),
    ("xgb", best_estimators[5]),
    ("lgbm", best_estimators[6])
], voting="soft", n_jobs=-1)

votingk = votingk.fit(X_resampled, y_resampled)
evaluate_model(votingk, X_resampled, y_resampled, X_test, y_test)

all_models = best_estimators + [votingk] + [votingC]
```
Evaluating on Train data
Accuracy: 0.9960655737704918
Precision: 0.9960962914769032
Recall: 0.9960655737704918
F1 Score: 0.9960655128657729
Kappa: 0.9921311475409836
Confusion Matrix:
 [[4575    0]
 [  36 4539]]


Evaluating on Test data
Accuracy: 0.6624242424242425
Precision: 0.7143910094057587
Recall: 0.6624242424242425
F1 Score: 0.6632864073294695
Kappa: 0.3486238922247359
Confusion Matrix:
 [[1071  241]
 [ 873 1115]]

This evaluation shows that the model performs exceptionally well on the training data. The accuracy, precision, recall, and F1 score are nearly 100%, and the Kappa score is 0.99, indicating that the model fits the training data very well and makes almost perfect predictions. The model has made almost no errors on the training data.
However, the model shows a significant drop in performance on the test data. The accuracy on the test data drops to 66.2%, while precision is 71.4%, recall is 66.2%, and the F1 score is 66.3%. The Kappa score is 0.35, indicating that the model has low agreement on the test data and struggles to generalize to real-world data.
The confusion matrix shows that the model can identify true positives (1115) well, but it makes serious errors with false positives (241) and false negatives (873). This indicates that the model is making too many misclassifications on the test data and shows signs of overfitting. The model has failed to generalize the patterns it learned from the training data to the test data.

```
votingJ = VotingClassifier(estimators=[
    ("dt", best_dt),
    ("rf", best_rf),
    ("knn", best_knn),
    ("xgb", best_xgb),
    ("lgbm", best_lgbm)
], voting="soft", n_jobs=-1)

votingJ = votingJ.fit(X_resampled, y_resampled)

evaluate_model(votingJ, X_resampled, y_resampled, X_test, y_test)

all_models = [best_dt, best_rf, best_knn, logreg_model, best_xgb, best_lgbm, votingJ]
```
Evaluating on Train data
Accuracy: 0.9083060109289618
Precision: 0.9222368071563409
Recall: 0.9083060109289618
F1 Score: 0.9075434092897675
Kappa: 0.8166120218579235
Confusion Matrix:
 [[4571    4]
 [ 835 3740]]


Evaluating on Test data
Accuracy: 0.6627272727272727
Precision: 0.7447609861467597
Recall: 0.6627272727272727
F1 Score: 0.6589729367065883
Kappa: 0.36376423471208386
Confusion Matrix:
 [[1167  145]
 [ 968 1020]]

This evaluation reveals that the model performs well on the training data, with high accuracy (90.8%), precision (92.2%), recall (90.8%), and F1 score (90.8%). The Kappa score of 0.82 indicates a strong agreement between the predicted and actual labels, suggesting that the model has learned the patterns in the training data effectively. The confusion matrix shows that the model correctly identifies most of the true positives and true negatives, with only a small number of misclassifications.
However, the performance on the test data is notably lower. The accuracy drops to 66.3%, with precision at 74.5%, recall at 66.3%, and an F1 score of 65.9%. The Kappa score also drops to 0.36, indicating a moderate level of agreement and suggesting that the model may be overfitting to the training data. The confusion matrix for the test data highlights that while the model still identifies a fair number of true positives, it struggles with false positives (145) and false negatives (968), leading to a decrease in overall predictive performance. This drop in performance from the training to the test data indicates that the model may have difficulty generalizing to new, unseen data.

Now let's combine the two balanced models, XGBClassifier and LGBMClassifier, and test them together.
```
votingX = VotingClassifier(estimators=[
    ("xgb", best_estimators[5]),
    ("lgbm", best_estimators[6])
], voting="soft", n_jobs=-1)

votingX = votingX.fit(X_resampled, y_resampled)

evaluate_model(votingX, X_resampled, y_resampled, X_test, y_test)

all_models = [best_estimators[5], best_estimators[6], votingX]
plot_model_performance(all_models, X_resampled, y_resampled, X_test, y_test)
```
At this point, the model we have obtained shows consistent performance on both training and test data. With an accuracy of 72.4% and an F1 score of around 0.70 on the training data, it demonstrates a reasonable ability to distinguish between classes. The test data performance is also similar, with an accuracy of 66.8% and an F1 score of 0.65, indicating that the model's generalization ability to the overall dataset is quite good.
Looking at the confusion matrix, we see a high number of correctly classified positive and negative instances in both training and test data, suggesting that the model effectively handles basic classification tasks.
In conclusion, this seems to be the best model we have obtained so far. Given the limitations of the dataset and current technical capabilities, this model stands out as a strong candidate in terms of both accuracy and generalization ability. However, our Kappa value is not as high as expected, which slightly undermines confidence in the model. To improve this, more data could be collected to enhance the model's learning, or we could optimize our hyperparameters over a broader range. Unfortunately, due to current technical constraints, this is not feasible at the moment. Therefore, this model appears to be the best solution for now, though there remains potential for future improvements.

![image](https://github.com/user-attachments/assets/7a1d6255-51e0-41d0-a826-2b9a916b9f23)

Overall, the XGBClassifier and LGBMClassifier models demonstrate strong performance, but the VotingClassifier's performance is nearly on par with these models. This suggests that the ensemble approach has the potential to maintain performance while enhancing generalization capabilities. By combining the strengths of various models, the VotingClassifier can provide more consistent and reliable predictions across a broader dataset.

## Testing Our Model With Postman
```
app = Flask(__name__)

model = joblib.load('votingX_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        input_data = np.array(data['input'])
        print(f"Input data: {input_data}")
        
        prediction = model.predict(input_data)
        print(f"Prediction: {prediction}")
  
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

def run_app():
    app.run(debug=True, use_reloader=False, port=9874)

thread = threading.Thread(target=run_app)
thread.start()
```
Warehouse_block,	Mode_of_Shipment,	Product_importance,	Discount_offered,	Weight_in_gms,	

#### Input
{
    "input": [
        [1, 1, 3, 10, 200],  // input 1: A, Ship, high, 10, 500
        [2, 2, 2, 5, 300],   // input 2: B, Flight, medium, 5, 300
        [3, 3, 1, 20, 400],  // input 3: C, Road, low, 20, 400
        [4, 1, 3, 15, 450]   // input 4: D, Ship, high, 15, 450
    ]
}

#### Output
{
    "prediction": [
        1,
        0,
        1,
        1
    ]
}

Reached on Time Distribution:
Comment: The target variable "Reached on Time" indicates whether the product reached its destination on time (0) or not (1). The data reveals that as the discount offered increases, the likelihood of the product being delivered on time also increases. This suggests that higher discounts may be linked to more efficient or prioritized shipping processes, which improves on-time delivery performance. This positive relationship between discount levels and timely delivery indicates that offering larger discounts might contribute to better delivery services, potentially by encouraging more effective logistics or customer service.
 
