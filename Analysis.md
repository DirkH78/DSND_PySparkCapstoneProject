# How to Predict Customer Churn in a Web Service with PySpark
## Motivation
As a provider of a web-based service, customer churn might have a major impact on the success of your service.
The good news is that you may already have acquired all the necessary data that can help you identify customers who are about to leave your service.
This is where machine learning and big data techniques can help you to detect critical customers and counteract with targeted countermeasures (nudging, proposals) before it might be to late.
In this article I would like to introduce a data science approach which can support you in this identification process and thus help you to maintain a popular service.
## Introduction
In this project we are dealing with server log data of a virtual music streaming platform calles "Sparkify".  
![Sparkify Logo](/bin/SparkLogo.jpg)  
The data set was kindly provided by the Udacity Data Scientist Nano-degree project team.
The size of server log files unfortunately wont allow us to perform a classical data analysis with pandas as many of us might be used to.
Data of this size might only be computable with clustered networks. Thus we should use Apache Spark (or PySpark respectively) to assess this amount of data.  
![Spark Logo](/bin/Spark-logo-192x100px.png)  
After importing the log file (json format) into a spark data frame the log entries include the following columns (among others):  
* __artist__: the songs artist
* __gender__: the users gender
* __level__: the subsribtion plan for the user (paid or free)
* __page__: the page a user is requesting
* __song__: the song a user has choosen
* __ts__: time stamp of the log entry
* __userId__: the user ID

Lets see how this information can help us to detect user churn.
But first lets concentrate on obvious informations that should be of general interest when it comes to identifying our customers wishes.
This data set alone allows us to identify first important characteristics of our service portfolio like:
##### our customers favorite artist:
```
df.filter(df.page == 'NextSong') \
  .select('Artist') \
  .groupBy('Artist') \
  .agg({'Artist':'count'}) \
  .withColumnRenamed('count(Artist)', 'Artistcount') \
  .sort(desc('Artistcount')) \
  .show(10)
  
+--------------------+-----------+
|              Artist|Artistcount|
+--------------------+-----------+
|       Kings Of Leon|       1841|
|            Coldplay|       1813|
|Florence + The Ma...|       1236|
|       Dwight Yoakam|       1135|
|               Björk|       1133|
|      The Black Keys|       1125|
|                Muse|       1090|
|       Justin Bieber|       1044|
|        Jack Johnson|       1007|
|              Eminem|        953|
+--------------------+-----------+
only showing top 10 rows
```
##### our customers favorite songs:
```
df.filter(df.page == 'NextSong') \
  .select('song', 'Artist') \
  .groupBy('song', 'Artist') \
  .agg({'song':'count'}) \
  .withColumnRenamed('count(song)', 'songcount') \
  .sort(desc('songcount')) \
  .show(10)
  
+--------------------+--------------------+---------+
|                song|              Artist|songcount|
+--------------------+--------------------+---------+
|      You're The One|       Dwight Yoakam|     1122|
|                Undo|               Björk|     1026|
|             Revelry|       Kings Of Leon|      854|
|       Sehr kosmisch|            Harmonia|      728|
|Horn Concerto No....|Barry Tuckwell/Ac...|      641|
|Dog Days Are Over...|Florence + The Ma...|      574|
|             Secrets|         OneRepublic|      463|
|        Use Somebody|       Kings Of Leon|      459|
|              Canada|    Five Iron Frenzy|      434|
|             Invalid|            Tub Ring|      424|
+--------------------+--------------------+---------+
only showing top 10 rows
```
But now lets concentrate on the "user churn"!
## Problem Statement
There are two major problems we will face in our approach to detect user churn:
* User churn is no variable that can directly be identified from our server logs. We need to derive feasable features that help us to identify user churn based on our existing data.
* The vast majority of users gratefully stay with our service. This is of course very favorable when we look at this fact from the perspective of user satisfaction. But this involves many statistical challenges when it comes to modelling user churn since our data is highly imbalanced.
## Considered Solutions
To adress these challenges I would like to propose the following methods:
* user churn: we will use a users page request for the "Cancellation Confirmation" to identify and define user churn.
* data imbalance / feasable metric: I would like to use the [F1 score](https://en.wikipedia.org/wiki/F1_score) to measure our models accuracy. The F1 score considers "recall" __and__ "precision" at the same time and thus is relatively protected against over-fitting of imbalanced data sets.
## Methodology
I would like to work through an established data analysis process first:
#### Data Cleaning
First we need to get rid of any relicts in our data set like entries with no user or session ID. I would also recommend to convert the user ID string to an integer value to ease the ordering process of the user IDs.
#### Data Exploration
Our initial analysis shows that we are dealing with 278154 individual log entries after cleaning.
We define "user churn" as a confirmed cancellation of a user account:
```
churn_users = df.filter(df.page == 'Cancellation Confirmation').select('userId').dropDuplicates()
churn_users_list = [user['userId'] for user in churn_users.collect()]
df = df.withColumn('churn', df.userId.isin(churn_users_list))
```
We can now identify the number of churned users:
```
df.filter(df.churn == True).select('userId').dropDuplicates().count()
```
Giving us only 52 users who have cancelled their accounts so far.
#### Feature Engineering
Since we are dealing with a certain user behaviour, our feature vector will provide different features for all registered users.
So lets calculate several features for all the individual users! One aspect that will be of importance for all the features is the content of our page column since here we can identify different actions our users perform.
Filtering for certain page accesses and grouping by the user helps us to come up with the following features:
* __userActivity__: how many actions has the user performed over the whole log file
* __SongsPlayed__: how many songs has the user played
* __NoOfThumbsUp__: how often has the user voted with a "Thumbs Up"
* __NoOfThumbsDown__: how often has the user voted with a "Thumbs Down"
* __NoOfErrors__: how many errors has the user faced
* __NoOfAddedFriends__: how many friends has the user added
* __NoOfSettingChanges__: how often did the user change the settings
* __deltaT__: how long was the users account active (this might look like "cheating" because we dont know the users activity within an online application. But this feature can be calculated online and helps to put a focus on new users who are most likely to leave our service.)
* __PaidFreeRatio__: what is the action step ratio of paid/free subscription for individual users who downgraded at least once (roughly estimated)
	* if the user never subscriped to full plan PaidFreeRatio is 0
	* if the user subscriped to full plan but never cancelled the PaidFreeRatio is 1
* __SongsPlayed_rel__: SongsPlayed divided by the time the users account was active
* __NoOfThumbsUp_rel__: NoOfThumbsUp divided by the time the users account was active
* __NoOfErrors_rel__: NoOfErrors divided by the time the users account was active
* __NoOfAddedFriends_rel__: NoOfAddedFriends divided by the time the users account was active  

At this point we can focus on some of the possible reasons why users leave our service by plotting 2 dimensional histograms which connect our features to possible user churn:  

|           feature 	| representation 	                                  | remarks                                                                                      |
|:--------------------	|:--------------------------------------------------: |--------------------------------------------------------------------------------------------: |
| no. of played songs 	|![Spark Logo](/bin/hist_playedSongs.png)             | users who churned are most likely to only play a limited amount of songs                     |
| no. of "thumbs up"  	|![Spark Logo](/bin/hist_thumbsUp.png)                | users who churned are most likely to only rate a limited amount of songs with a "thumbs up"  |
| no. of errors      	|![Spark Logo](/bin/hist_error.png)                   | users who churned often faced one major error before leaving the service                     |

#### Modelling
For the modelling process I selected three different models which were integreted into a machine learning pipeline (scaling, normalizing, indexing).
The modelling process involves many challanges like over-fitting and a lack of generalization optins of the model within future applications with unknown data. To adress these challanges all model pipelines were integrated into a kFold cross validation with several diffenrent hyperparmaters for the model otions. This allows the identification of the best suited model and model options for the task at hand.  
In this case we used the following hyperparameter variations:
##### Logistic Regression Variations
* lambda value of [regularization](https://runawayhorse001.github.io/LearningApacheSpark/reg.html):
	* 0.0, 0.1, 0.01, 0.005
##### Decision Tree Variations
* impurity measures:
	* entropy, gini
* maximal depth:
	* 0, 1, 2, 3, 4, 5
##### Random Forest Variations
* number of trees:
	* 10, 30  

After training (cross-validation) the pipeline with all several different hyperparameter sets, I could observe the following "best" performances (F1-score):  

|               Model 	| Training Performance 	| Test Performance 	| Best Hyperparameters |
|:--------------------	|---------------------:	|----------------:	|-------------------:  |
| Logistic Regression 	|              89.32 % 	|         91.52 % 	| lambda value = 0.0   |
|       Decision Tree 	|              94.55 % 	|         92.76 % 	| maximal depth = 1    |
|       Random Forest 	|              92.11 % 	|         92.76 % 	| number of trees = 30 |

## Conclusion
When it comes to customer churn, a decision tree model (with a maximal depth of 1) with the features mentioned above gives a reliable indication of whether a customer is about to cancel the service.
Integrating this pipeline in your web application might help you to detect a critical customer in advance and provide suitable countermeasures.
## Robustness
Since we are facing highly imbalanced and reduced data, the robustness was validated by changing the so called "seed value". This value affects random choices within python (train-test-split, starting conditions,...). Changing the seed value showed an influence on the modelling accuracy that can not be fully neglected. To adress this issue, utilizing a data set with significantly more entries would be advisable!  
## Reflections and Improvement
This approach might be optimized by including further state transitions (like "PaidFreeRatio") into the feature vector or adding further hyperparameters into the modelling pipelines.
For me this little project was extremly valuable to get into PySpark and to widen my data science horizon.