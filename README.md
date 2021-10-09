# Exploratory data analysis on haberman's survival dataset
# INTRODUCTION
Haberman’s data set contains data from the study conducted in University of Chicago’s Billings Hospital between year 1958 to 1970 for the patients who undergone surgery of breast cancer. 
# SOURCE
https://www.kaggle.com/gilsousa/habermans-survival-data-set
# DESCRIPTION
I would like to explain the various data analysis operation, I have done on this data set and how to conclude or predict survival status of patients who undergone from surgery.


First of all for any data analysis task or for performing operation on data we should have good domain knowledge so that we can relate the data features and also can give accurate conclusion. So, I would like to explain the features of data set and how it affects other feature.


There are 4 attribute in this data set out of which 3 are features and 1 class attribute as below. Also, there are 306 instances of data.

1.Number of Axillary nodes(Lymph Nodes)

2.Age

3.Operation Year

4.Survival Status


Lymph Node: Lymph nodes are small, bean-shaped organs that act as filters along the lymph fluid channels. As lymph fluid leaves the breast and eventually goes back into the bloodstream, the lymph nodes try to catch and trap cancer cells before they reach other parts of the body. Having cancer cells in the lymph nodes under your arm suggests an increased risk of the cancer spreading.In our data it is axillary nodes detected(0–52).
(Source: https://www.breastcancer.org/symptoms/diagnosis/lymph_nodes)

Age: It represent the age of patient at which they undergone surgery (age from 30 to 83)

Operation year: Year in which patient was undergone surgery(1958–1969).

Survival Status: It represent whether patient survive more than 5 years or less after undergone through surgery.Here if patients survived 5 years or more is represented as 1 and patients who survived less than 5 years is represented as 2.

So lets get started 

Importing important libraries to be used for our analysis 

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
```

Reading dataset to be used for analysis
```python
haberman = pd.read_csv("haberman (2).csv")
```
this dataset consist of total list of 306 patients that classified on basis of their age,year of operation ,nodes and status .

status has numerical value of either 1 (that represents patients survive 5 years or more after operation) and 2 (that represents patients survive 5 years or less).

```python
haberman["status"].value_counts()
```
1    225

2     81

Name: status, dtype: int64

In dataset we have total of 225 number of patients with survival years or more and 81 no of patients with survival years less than 5 years.

```python
sns.FacetGrid(haberman, hue="status", size=4) \
   .map(plt.scatter, "age", "year") \
   .add_legend();
plt.show();
```
Above plot uses seaborn library in which function facetgrid is used whicg uses different colors for both the status taking in account two variables age, year.

![Screen Shot 2021-10-09 at 4 40 19 PM](https://user-images.githubusercontent.com/90976062/136655742-a39e1d1c-e396-4d4b-9f84-8608b4d05987.png)


Though from the above plot we are not able to come to any conclusion as both features used show mixed and scattered data .

Now we are again usung seaborn library to create a pairplot in order to come up with pairs of features giving use much better analysis 

![Screen Shot 2021-10-09 at 4 46 41 PM](https://user-images.githubusercontent.com/90976062/136655860-25efafea-083c-42eb-97bd-d2299127d9ae.png)

From above pairplots we find age and nodes to most suitable to analyse the data furthur as it has least overlapping compared to others


```python
import numpy as np
haberman_1 = haberman.loc[haberman["status"] == 1]; 
haberman_2 = haberman.loc[haberman["status"] == 2];
plt.plot(haberman_1["nodes"], np.zeros_like(haberman_1['nodes']), 'o')
plt.plot(haberman_2["nodes"], np.zeros_like(haberman_2['nodes']), 'o')
#plt.plot(iris_virginica["petal_length"], np.zeros_like(iris_virginica['petal_length']), 'o')

plt.show()
```

In above code we are using numpy library to plot a 1-D scatter plot showing status.

![Screen Shot 2021-10-09 at 5 16 20 PM](https://user-images.githubusercontent.com/90976062/136656697-5891a004-b5a4-4144-9976-7778fa8588cd.png)

Above plot uses only one feature node 

we are not able to make any conclusions from 1-D scatter plot as most of points are overlapping each other 

Now we are going to use distribution plots that are part of seaborn library to get univariante analysis for each of the feature

```python
sns.FacetGrid(haberman, hue="status", size=5) \
   .map(sns.distplot, "nodes") \
   .add_legend();
plt.show();
```

![Screen Shot 2021-10-09 at 5 23 48 PM](https://user-images.githubusercontent.com/90976062/136656858-ed23a0f4-f6c5-455c-8441-0dc7af659ce3.png)

From above plot we can say patients with less number of nodes tend to survive more patients with node size = 0 or closer to it having much higher change of surviving 5 years or more

```python
sns.FacetGrid(haberman, hue="status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();
```

![Screen Shot 2021-10-09 at 5 33 53 PM](https://user-images.githubusercontent.com/90976062/136657124-511e7825-8ca3-4655-9be5-57ff990a8879.png)

From this plot though there is no clear saturation of age still we can make a rough prediction as 

 if age <35 patients usually survive more than 5 years  
 elseif age > 75 patients usually survive less than 5 years 
 
 ```python
 sns.FacetGrid(haberman, hue="status", size=5) \
   .map(sns.distplot, "year") \
   .add_legend();
plt.show();
```

![Screen Shot 2021-10-09 at 5 37 57 PM](https://user-images.githubusercontent.com/90976062/136657235-c48cae2a-d119-46cf-81d6-6b266f89c272.png)

Cannot make any conclusion from this distribution plot as it is overlapping at every point.

Let’s plot CDF for our selected feature which is Axillary nodes

```python
counts, bin_edges = np.histogram(haberman_1['nodes'], bins=10, 
                                 density = True)
print(counts)
pdf = counts/(sum(counts))

print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();
```

![Screen Shot 2021-10-09 at 5 52 46 PM](https://user-images.githubusercontent.com/90976062/136657763-b1d35903-ed60-4cc0-bdac-24ca54ce119a.png)

From above CDF you can observe that orange line shows there is a 85% chance of long survival if number of axillary nodes detected are < 5.

Getting medians for status = 1 and status = 2 with nodes as variable 

```python
print("\nMedians:")
print(np.median(haberman_1["nodes"]))
#Median with an outlier
print(np.median(np.append(haberman_1["nodes"],50)));
print(np.median(haberman_2["nodes"]))
```

 average node size of patients with survival more than 5 years is 0 .
 
 average node size of patients with survival less than 5 years is 4.
 
 Now using seaborn library again to plot a boxplot , boxplots effectiviely visualize data on basis of 25% , 50% , 75% that comes handy most of the times 
 
 Lets see
 
 ```python
 sns.boxplot(x='status',y='nodes', data=haberman)
plt.show()
```
![Screen Shot 2021-10-09 at 6 15 30 PM](https://user-images.githubusercontent.com/90976062/136658400-3106cf94-ad9b-4d6b-a2a5-bc108ce0b74d.png)

 from box plot wecan easily identify 25-50-75 percentile of values .
 
 75 % of patients with survival more than 5 years have node size <= 2
 
 only 25 % of patients with survival less than 5 years have node size <=1
 
 and 75 % of patients with survival less than 5 years have node size <=11
 
 50 % of pateints with survival less than 5 years have node size <=5
 
 
