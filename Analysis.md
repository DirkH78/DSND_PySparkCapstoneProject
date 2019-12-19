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
## Methodology
I would like to work through an established data analysis process:
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
* __deltaT__: how long was the users account active
* __PaidFreeRatio__: what is the action step ratio of paid/free subscription for individual users who downgraded at least once (roughly estimated)
	* if the user never subscriped to full plan PaidFreeRatio is 0
	* if the user subscriped to full plan but never cancelled the PaidFreeRatio is 1
* __SongsPlayed_rel__: SongsPlayed divided by the time the users account was active
* __NoOfThumbsUp_rel__: NoOfThumbsUp divided by the time the users account was active
* __NoOfErrors_rel__: NoOfErrors divided by the time the users account was active
* __NoOfAddedFriends_rel__: NoOfAddedFriends divided by the time the users account was active
#### Modelling
For the modelling process I selected three different models which were integreted into a machine learning pipeline (scaling, normalizing, indexing). After training I could realize the following performances provided by a binary classification evaluator:  

|               Model 	| Training Performance 	| Test Performance 	|
|:--------------------	|---------------------:	|----------------:	|
| Logistic Regression 	|              90.42 % 	|         66.07 % 	|
|       Decision Tree 	|              89.52 % 	|         75.00 % 	|
|       Random Forest 	|              97.83 % 	|         83.04 % 	|

## Conclusion
When it comes to customer churn, a random forest model (10 trees) with the features mentioned above gives a reliable indication of whether a customer is about to cancel the service.
Integrating this pipeline in your web application might help you to detect a critical customer in advance and provide suitable countermeasures.
## Reflections and Improvement
This approach might be optimized by including further state transitions (like "PaidFreeRatio") into the feature vector or adding further hyperparameters into the modelling pipelines.
For me this little project was extremly valuable to get into PySpark and to widen my data science horizon.