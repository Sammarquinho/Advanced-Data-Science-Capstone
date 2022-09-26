# Advanced-Data-Science-Capstone

IBM Coursera Advanced Data Science Capstone

Name	Description	Date

Daniel Sammarco	IBM Coursera Advanced Data Science Capstone.	on 26th of September 2022

Building an artist recommender system

As its name implies, a recommender system is a tool that helps predicting what a user may or may not like among a list of given items. In some sense, you can view this as an alternative to content search, as recommendation engines help users discover products or content that they may not come across otherwise. For example, Facebook suggests friends and pages to users. Youtube recommends videos which users may be interested in. Amazon suggests the products which users may need... Recommendation engines engage users to services, can be seen as a revenue optimization process, and in general help maintaining interest in a service.

In this notebook, I demonstrate how to build a simple recommender system: we focus on music recommendations, and we use a simple algorithm to predict which items users might like, that is called ALS -- alternating least squares.

Goals
In this notebook, I expect to:

Revisit (or learn) recommender algorithms

Understand the idea of Matrix Factorization and the ALS algorithm (serial and parallel versions)

Build a simple model for a real usecase: music recommender system

Understand how to validate the results

Steps
Inspect the data using Spark SQL, and build some basic, but very valuable knowledge about the information we have at hand
Formally define what is a sensible algorithm to achieve our goal: given the "history" of user taste for music, recommend new music to discover. Essentialy, we want to build a statistical model of user preferences such that we can use it to "predict" which additional music the user could like
With our formal definition at hand, we will learn different ways to implement such an algorithm. Our goal here is to illustrate what are the difficulties to overcome when implementing a (parallel) algorithm
Finally, we will focus on an existing implementation, available in the Apache Spark MLLib, which we will use out of the box to build a reliable statistical model
Furthermore:

One important topic that I will cover in the Notebooks is how to validate the results we obtain, and how to choose good parameters to train models especially when using an "opaque" library for doing the job. As a consequence, I will focus on the statistical validation of our recommender system.

1. Data
Understanding data is one of the most important part when designing any machine learning algorithm. In this notebook, I will use a data set published by Audioscrobbler - a music recommendation system for last.fm. Audioscrobbler is also one of the first internet streaming radio sites, founded in 2002. It provided an open API for “scrobbling”, or recording listeners’ plays of artists’ songs. last.fm used this information to build a powerful music recommender engine.

1.1. Data schema
Unlike a rating dataset which contains information about users' preference for products (one star, 3 stars, and so on), the datasets from Audioscrobbler only has information about events: specifically, it keeps track of how many times a user played songs of a given artist and the names of artists. That means it carries less information than a rating: in the literature, this is called explicit vs. implicit ratings.

Reading material
Implicit Feedback for Inferring User Preference: A Bibliography
Comparing explicit and implicit feedback techniques for web retrieval: TREC-10 interactive track report
Probabilistic Models for Data Combination in Recommender Systems
The data we use in this Notebook is available in 3 files (these files are stored in our HDFS layer, in the directory /dataset/):

user_artist_data.txt: It contains about 140,000+ unique users, and 1.6 million unique artists. About 24.2 million users’ plays of artists’ are recorded, along with their count. It has 3 columns separated by spaces:
UserID	ArtistID	PlayCount
...	...	...
artist_data.txt : It prodives the names of each artist by their IDs. It has 2 columns separated by tab characters (\t).
ArtistID	Name
...	...
artist_alias.txt: Note that when plays are scrobbled, the client application submits the name of the artist being played. This name could be misspelled or nonstandard. For example, "The Smiths", "Smiths, The", and "the smiths" may appear as distinct artist IDs in the data set, even though they are plainly the same. artist_alias.txt maps artist IDs that are known misspellings or variants to the canonical ID of that artist. The data in this file has 2 columns separated by tab characters (\t).
MisspelledArtistID	StandardArtistID
...	...
1.2. Data Exploration: simple descriptive statistics
In order to choose or design a suitable algorithm for achieving our goals, given the data we have, we should first understand data characteristics. To start, we import the necessary packages to work with regular expressions, Data Frames, and other nice features of our programming environment.

# Downloading dataset
!set -e

!mkdir -p dataset
!cd dataset
!wget http://www.iro.umontreal.ca/~lisa/datasets/profiledata_06-May-2005.tar.gz
!tar xvf profiledata_06-May-2005.tar.gz
!mv profiledata_06-May-2005/* dataset/
!rm -r profiledata_06-May-2005
--2018-11-23 06:24:25--  http://www.iro.umontreal.ca/~lisa/datasets/profiledata_06-May-2005.tar.gz
Resolving www.iro.umontreal.ca (www.iro.umontreal.ca)... 132.204.26.36
Connecting to www.iro.umontreal.ca (www.iro.umontreal.ca)|132.204.26.36|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 135880312 (130M) [application/x-gzip]
Saving to: ‘profiledata_06-May-2005.tar.gz.2’

100%[======================================>] 135,880,312 25.1MB/s   in 4.8s   

2018-11-23 06:24:31 (27.2 MB/s) - ‘profiledata_06-May-2005.tar.gz.2’ saved [135880312/135880312]

profiledata_06-May-2005/
profiledata_06-May-2005/artist_data.txt
profiledata_06-May-2005/README.txt
profiledata_06-May-2005/user_artist_data.txt
profiledata_06-May-2005/artist_alias.txt
!cat dataset/user_artist_data.txt | head
1000002 1 55
1000002 1000006 33
1000002 1000007 8
1000002 1000009 144
1000002 1000010 314
1000002 1000013 8
1000002 1000014 42
1000002 1000017 69
1000002 1000024 329
1000002 1000025 1
cat: write error: Broken pipe
import os
import sys
import re
import random
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.functions import *

%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time

sqlContext = SQLContext(sc)
# Base directory path
base = "dataset/"
Using SPARK SQL, load data from /dataset/user_artist_data.txt and show the first 20 entries (via function show()).

For this Notebook, from a programming point of view, we are given the schema for the data we use, which is as follows:

userID: long int
artistID: long int
playCount: int
Each line of the dataset contains the above three fields, separated by a "white space".

userArtistDataSchema = StructType([ \
    StructField("userID", LongType(), True), \
    StructField("artistID", LongType(), True), \
    StructField("playCount", IntegerType(), True)])

userArtistDF = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .options(header='false', delimiter=' ') \
    .load(base + "user_artist_data.txt", schema = userArtistDataSchema) \
    .cache()

# we can cache an Dataframe to avoid computing it from the beginning everytime it is accessed.
userArtistDF.cache()

userArtistDF.show()
+-------+--------+---------+
| userID|artistID|playCount|
+-------+--------+---------+
|1000002|       1|       55|
|1000002| 1000006|       33|
|1000002| 1000007|        8|
|1000002| 1000009|      144|
|1000002| 1000010|      314|
|1000002| 1000013|        8|
|1000002| 1000014|       42|
|1000002| 1000017|       69|
|1000002| 1000024|      329|
|1000002| 1000025|        1|
|1000002| 1000028|       17|
|1000002| 1000031|       47|
|1000002| 1000033|       15|
|1000002| 1000042|        1|
|1000002| 1000045|        1|
|1000002| 1000054|        2|
|1000002| 1000055|       25|
|1000002| 1000056|        4|
|1000002| 1000059|        2|
|1000002| 1000062|       71|
+-------+--------+---------+
only showing top 20 rows

How many distinct users do we have in our data?
allusers = userArtistDF.count()
print("All rows in database: ", allusers )
uniqueUsers = userArtistDF.select('userID').distinct().count()
print("Total n. of distinct users: ", uniqueUsers)
All rows in database:  24296858
Total n. of distinct users:  148111
How many distinct artists do we have in our data ?
uniqueArtists = userArtistDF.select('artistID').distinct().count()
print("Total n. of artists: ", uniqueArtists)
Total n. of artists:  1631028
What are the maximum and minimum values of column userID ?
One limitation of Spark MLlib's ALS implementation - which we will use later - is that it requires IDs for users and items to be nonnegative 32-bit integers. This means that IDs larger than Integer.MAX_VALUE, or 2147483647, can't be used. So we need to check whether this data set conforms to the strict requirements of our library.

MAX_VALUE = 2147483647
#showing the IDs which are invalid
userArtistDF[(userArtistDF.userID.cast("int") < 0)].show()
userArtistDF[(userArtistDF.userID.cast("int") > MAX_VALUE)].show()
#As the result, we don't see any invalid userID
#Otherwise, we can use function describe() of Spark MLlib to show the statistics
userArtistDF.select("userID").describe().show()
+------+--------+---------+
|userID|artistID|playCount|
+------+--------+---------+
+------+--------+---------+

+------+--------+---------+
|userID|artistID|playCount|
+------+--------+---------+
+------+--------+---------+

+-------+------------------+
|summary|            userID|
+-------+------------------+
|  count|          24296858|
|   mean|1947573.2653533637|
| stddev| 496000.5551820078|
|    min|                90|
|    max|           2443548|
+-------+------------------+

What is the maximum and minimum values of column artistID ?
#Finding min and max of artistID using function groupby() and max|min()
max_artistID = userArtistDF.groupby().max('artistID').collect()[0].asDict()['max(artistID)']
print('Maximum value of artistID: ',max_artistID)
min_artistID = userArtistDF.groupby().min('artistID').collect()[0].asDict()['min(artistID)']
print('Minimum value of artistID: ',min_artistID)

#Again, we can use describe() for short
userArtistDF.select("artistID").describe().show()
Maximum value of artistID:  10794401
Minimum value of artistID:  1
+-------+------------------+
|summary|          artistID|
+-------+------------------+
|  count|          24296858|
|   mean|1718704.0937568964|
| stddev|2539389.0924284603|
|    min|                 1|
|    max|          10794401|
+-------+------------------+

We just discovered that we have a total of 148,111 users in our dataset. Similarly, we have a total of 1,631,028 artists in our dataset. The maximum values of userID and artistID are still smaller than the biggest number of integer type. No additional transformation will be necessary to use these IDs.

One thing we can see here is that SPARK SQL provides very concise and powerful methods for data analytics (compared to using RDD and their low-level API). You can see more examples here.

Next, we might want to understand better user activity and artist popularity.

Here is a list of simple descriptive queries that helps us reaching these purposes:

How many times each user has played a song? This is a good indicator of who are the most active users of our service. Note that a very active user with many play counts does not necessarily mean that the user is also "curious"! Indeed, she could have played the same song several times.
How many play counts for each artist? This is a good indicator of the artist popularity. Since we do not have time information associated to our data, we can only build a, e.g., top-10 ranking of the most popular artists in the dataset. Later in the notebook, we will learn that our dataset has a very "loose" definition about artists: very often artist IDs point to song titles as well. This means we have to be careful when establishing popular artists. Indeed, artists whose data is "well formed" will have the correct number of play counts associated to them. Instead, artists that appear mixed with song titles may see their play counts "diluted" across their songs.
How many times each user has played a song? Display 5 samples of the result.
# Compute user activity
# We are interested in how many playcounts each user has scored.
userActivity = userArtistDF.groupBy('userID').sum('playCount').collect()
print(userActivity[0:5])
len(userActivity)
[Row(userID=1000061, sum(playCount)=244), Row(userID=1000070, sum(playCount)=20200), Row(userID=1000313, sum(playCount)=201), Row(userID=1000832, sum(playCount)=1064), Row(userID=1000905, sum(playCount)=214)]
148111
Plot CDF (or ECDF) of the number of play counts per User ID.
Look at important percentiles (25%, median, 75%, tails such as >90%)
Discuss about users: we will notice that for some users, there is very little interaction with the system, which means that maybe reccommending something to them is going to be more difficult than for other users who interact more with the system.
Look at outliers and reasons about their impact on the reccommender algorithm
pdf = pd.DataFrame(data=userActivity)
Y=np.sort( pdf[1] )
yvals=np.arange(len(Y))/float(len(Y))

print(np.arange(len(Y)))

plt.semilogx( Y, yvals )
plt.xlabel('Play Counts')
plt.ylabel('ECDF')
plt.grid(True,which="both",ls="-")
plt.title('ECDF of number of play counts per User ID')
plt.show()

#Morever, I would like to show some statistical measurements of userAcitvity
print ('Total =', Y.sum())
print ('Mean =', Y.mean())
print ('Min =', Y.min())
print ('Max =', Y.max())

# look at important percentiles (25%, median, 75%, tails such as >90%)
print('Percentile 25% :' + str(np.percentile(Y,25)))
print('Percentile 50% :' + str(np.percentile(Y,50)))
print('Percentile 75% :' + str(np.percentile(Y,75)))
print('Percentile 90% :' + str(np.percentile(Y,90)))
print('Percentile 95% :' + str(np.percentile(Y,95)))
print('Percentile 99% :' + str(np.percentile(Y,99)))

# look at the percentile has playCount less than 10
print('The percentage of user playing less than 10 times P(Y<=10) =', len(Y[Y<=10])/len(Y))
[     0      1      2 ..., 148108 148109 148110]

Total = 371638969
Mean = 2509.1922207
Min = 1
Max = 674412
Percentile 25% :204.0
Percentile 50% :892.0
Percentile 75% :2800.0
Percentile 90% :6484.0
Percentile 95% :10120.0
Percentile 99% :21569.2
The percentage of user playing less than 10 times P(Y<=10) = 0.05228511049145573
We have total 371638969 play counts.
In average, every user plays 2509 times.
25% of the users have the play counts less than or equal to (<=) 204 times.
50% of the users have the play counts less than or equal to (<=) 892times.
75% of the users have the play counts less than or equal to (<=) 2800 times.
95% of the users have the play counts less than or equal to (<=) 10120 times.
99% of the users have the play counts less than or equal to (<=) 21569 times.
The result is plausible with the figure above.
About 7746 users (5.23%) have the play counts less than or equal to (<=) 10 times. These users have very little interaction with the system, so there is more difficult for recommending for these users than other users creating more impact in the system (have a certain number of playCount).
How many play counts for each artist?
# Compute artist popularity
# We are interested in how many playcounts per artist
# ATTENTION! Grouping by artistID may be problematic, as stated above.

artistPopularity = userArtistDF.groupBy('artistID').sum('playCount').collect()
print(artistPopularity[0:5]) #print first 5 artistID in dataframe
len(artistPopularity)
[Row(artistID=1003514, sum(playCount)=949), Row(artistID=1004346, sum(playCount)=3772), Row(artistID=5409, sum(playCount)=526693), Row(artistID=1002519, sum(playCount)=405), Row(artistID=1004223, sum(playCount)=409)]
1631028
pdf1 = pd.DataFrame(data=artistPopularity)
Y1=np.sort( pdf1[1] )
yvals1=np.arange(len(Y1))/float(len(Y1))

print(np.arange(len(Y1)))

plt.semilogx( Y1, yvals1 )
plt.xlabel('Play Counts')
plt.ylabel('ECDF')
plt.grid(True,which="both",ls="-")
plt.title('ECDF of number of play counts per artist ID')
plt.show()



#
print ('Sum =', Y1.sum())
print ('Mean =', Y1.mean())
print ('Min =', Y1.min())
print ('Max =', Y1.max())
print ('Top 5 play counts:', Y1[len(Y1)-5:len(Y1)])
print ('Sum top 5 artist play counts:', Y1[len(Y1)-5:len(Y1)].sum())
print ('Percentage of top 5 artist play counts:', Y1[len(Y1)-5:len(Y1)].sum()/Y1.sum())
print ('P(playCount<=10) =', len(Y1[Y1<=10])/len(Y1))
print ('P(playCount<=1000) =', len(Y1[Y1<=1000])/len(Y1))
[      0       1       2 ..., 1631025 1631026 1631027]

Sum = 371638969
Mean = 227.855664648
Min = 1
Max = 2502130
Top 5 play counts: [1425942 1542806 1930592 2259185 2502130]
Sum top 5 artist play counts: 9660655
Percentage of top 5 artist play counts: 0.0259947309239
P(playCount<=10) = 0.7486793605014751
P(playCount<=1000) = 0.987435531456235
Answer :
Total 371638969 play counts. In average, playCount per artist is 227 times
Only 74.87% of the artists is played less than or equal to (<=) 10 times.
And 98.74% of the artists is played less than or equal to (<=) 1000 times.
In the other hand, we have top 5 artist play counts: [1425942 1542806 1930592 2259185 2502130]. This accounts for 2.6% on overall number of playCount (5 out of 1631028 artists). Moreover, the play count of top 5 artist is much higher than the mean. So we can implie that we can recommend most-played artists to every user with this top 5 artirst, and still get high performance.
Plot a bar chart to show top 5 artists In terms of absolute play counts.
Are these reasonable results?
Looking at top-5 artists
Anything strange in the data?
sortedArtist = sorted(artistPopularity, key = lambda x: -x[1])[:5]

artistID = [w[0] for w in sortedArtist]

y_pos = range(len(sortedArtist))
frequency = [w[1] for w in sortedArtist]

plt.barh(y_pos, frequency[::-1], align='center', alpha=0.4)
plt.yticks(y_pos, artistID[::-1])
plt.xlabel('Play Count')
plt.ylabel('Artist')
plt.title('Top-5 Artist ID per play counts')
plt.show()

Answer:
This resuls seem not reasonable.
In the previous, 98.74% of the artists have the play count less than or equal to (<=) 1000 times. So, top-5-artists is not sufficient to extract infomation about the data, they are the outliers . We should cut them out of data.
As the result, top-5-artists take about 2.6% of the total play counts, but 98.74% of the artists have the play count less than or equal to (<=) 1000 times. Then this is obviously that some artists are played much more than other artists.
All seems clear right now, but ... wait a second! What about the problems indicated above about artist "disambiguation"? Are these artist ID we are using referring to unique artists? How can we make sure that such "opaque" identifiers point to different bands? Let's try to use some additional dataset to answer this question: artist_data.txt dataset. This time, the schema of the dataset consists in:

artist ID: long int
name: string
We will try to find whether a single artist has two different IDs.

1.3 Data Integration
Loading artist data
Load the data from /dataset/artist_data.txt and use the SparkSQL API to show 5 samples.

customSchemaArtist = StructType([ \
    StructField("artistID", LongType(), True), \
    StructField("name", StringType(), True)])

artistDF = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .options(header='false', delimiter='\t', mode='DROPMALFORMED') \
    .load(base + "artist_data.txt", schema = customSchemaArtist) \
    .cache()
artistDF.show(5)
+--------+--------------------+
|artistID|                name|
+--------+--------------------+
| 1134999|        06Crazy Life|
| 6821360|        Pang Nakarin|
|10113088|Terfel, Bartoli- ...|
|10151459| The Flaming Sidebur|
| 6826647|   Bodenstandig 3000|
+--------+--------------------+
only showing top 5 rows

Look at top 20 artists whose name contains "Aerosmith". Take a look at artists that have ID equal to 1000010 and 2082323. Are they pointing to the same artist?
HINT: Function locate(sub_string, string) can be useful in this case.

# get artists whose name contains "Aerosmith"
artistDF[locate("Aerosmith", artistDF.name) > 0].show(20, False)

# show two examples
artistDF[artistDF.artistID==1000010].show()
artistDF[artistDF.artistID==2082323].show()
+--------+----------------------------------------------+
|artistID|name                                          |
+--------+----------------------------------------------+
|10586006|Dusty Springfield/Aerosmith                   |
|6946007 |Aerosmith/RunDMC                              |
|10475683|Aerosmith: Just Push Play                     |
|1083031 |Aerosmith/ G n R                              |
|6872848 |Britney, Nsync, Nelly, Aerosmith,Mary J Blige.|
|10586963|Green Day - Oasis - Eminem - Aerosmith        |
|10028830|The Aerosmith Antology2                       |
|10300357|Run-DMC + Aerosmith                           |
|2027746 |Aerosmith by MusicInter.com                   |
|1140418 |[rap]Run DMC and Aerosmith                    |
|10237208|Aerosmith + Run DMC                           |
|10588537|Aerosmith, Kid Rock, & Run DMC                |
|9934757 |Aerosmith - Big Ones                          |
|10437510|Green Day ft. Oasis & Aerosmith               |
|6936680 |RUN DNC & Aerosmith                           |
|10479781|Aerosmith Hits                                |
|10114147|Charlies Angels - Aerosmith                   |
|1262439 |Kid Rock, Run DMC & Aerosmith                 |
|7032554 |Aerosmith & Run-D.M.C.                        |
|10033592|Aerosmith?                                    |
+--------+----------------------------------------------+
only showing top 20 rows

+--------+---------+
|artistID|     name|
+--------+---------+
| 1000010|Aerosmith|
+--------+---------+

+--------+------------+
|artistID|        name|
+--------+------------+
| 2082323|01 Aerosmith|
+--------+------------+

In my opinion, they are pointing to the same artist
To answer this question correctly, we need to use an additional dataset artist_alias.txt which contains the ids of mispelled artists and standard artists. The schema of the dataset consists in:

mispelledID ID: long int
standard ID: long int
Using SparkSQL API, load the dataset from /dataset/artist_alias.txt then show 5 samples.

customSchemaArtistAlias = StructType([ \
    StructField("mispelledID", LongType(), True), \
    StructField("standardID", LongType(), True)])

artistAliasDF = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .options(header='false', delimiter='\t', mode='DROPMALFORMED') \
    .load(base + "artist_alias.txt", schema = customSchemaArtistAlias) \
    .cache()

artistAliasDF.show(5)
+-----------+----------+
|mispelledID|standardID|
+-----------+----------+
|    1092764|   1000311|
|    1095122|   1000557|
|    6708070|   1007267|
|   10088054|   1042317|
|    1195917|   1042317|
+-----------+----------+
only showing top 5 rows

Verify the answer of "Are artists that have ID equal to 1000010 and 2082323 the same ?" by finding the standard ids corresponding to the mispelled ids 1000010 and 2082323 respectively.
artistAliasDF[artistAliasDF.mispelledID == "1000010" ].show()
artistAliasDF[artistAliasDF.mispelledID == "2082323" ].show()

# 1000010 is a standard id, so it haven't been considered as mispelled id in the dataset
+-----------+----------+
|mispelledID|standardID|
+-----------+----------+
+-----------+----------+

+-----------+----------+
|mispelledID|standardID|
+-----------+----------+
|    2082323|   1000010|
+-----------+----------+

After evaluating mispelledID, my guess is correct because the artistID = "2082323" has standardID="1000010", which means representing the same artist
The mispelled or nonstandard information about artist make our results in the previous queries a bit "sloppy". To overcome this problem, we can replace all mispelled artist ids by the corresponding standard ids and re-compute the basic descriptive statistics on the "amended" data. First, we construct a "dictionary" that maps non-standard ids to a standard ones. Then this "dictionary" will be used to replace the mispelled artists.

From data in the dataframe loaded from /dataset/artist_alias.txt, construct a dictionary that maps each non-standard id to its standard id.

artistAlias = artistAliasDF.rdd.map(lambda row: (row.mispelledID,row.standardID)).collectAsMap()

#checking the total number of standard artistID
len(artistAlias)
190893
The total number row of the dictionary now 190893 artists. This gives me the lesson that we should standalize the field of the data which more likely to create ambiguity
Using the constructed dictionary artistAlias, replace the non-standard artist ids in the dataframe that was loaded from /dataset/user_artist_data.txt by the corresponding standard ids then show 5 samples.

NOTE: If an id doesn't exist in the dictionary as a mispelled id, it is really a standard id.

Using funtion map on Spark Dataframe will give us an RDD. We can convert this RDD back to Dataframe by using sqlContext.createDataFrame(rdd_name, sql_schema)

from time import time

bArtistAlias = sc.broadcast(artistAlias)

def replaceMispelledIDs(fields):
    finalID = bArtistAlias.value.get(fields[1] ,fields[1])
    return (fields[0], finalID, fields[2])

t0 = time()

newUserArtistDF = sqlContext.createDataFrame(
    userArtistDF.rdd.map(replaceMispelledIDs), 
    userArtistDataSchema
)
newUserArtistDF.show(5)
t1 = time()

print('The script takes %f seconds' %(t1-t0))
newUserArtistDF = newUserArtistDF.cache()
+-------+--------+---------+
| userID|artistID|playCount|
+-------+--------+---------+
|1000002|       1|       55|
|1000002| 1000006|       33|
|1000002| 1000007|        8|
|1000002| 1000009|      144|
|1000002| 1000010|      314|
+-------+--------+---------+
only showing top 5 rows

The script takes 0.377579 seconds
Although having some advantages, explicitly creating broadcast variables is only useful when tasks across multiple stages need the same data or when caching the data in deserialized form is important.

Well, our data frame contains clean and "standard" data. We can use it to redo previous statistic queries.

How many unique artists? Compare with the result when using old data.
uniqueArtists = newUserArtistDF.select('artistID').distinct().count()

print("Total n. of artists: ", uniqueArtists)
Total n. of artists:  1568126
The number of artists is reduced (from 1631028 to 1568126) after cleaning and standardization (replacing mispelledID with standardID).
Who are the top-10 artistis?
In terms of absolute play counts
In terms of "audience size", that is, how many users listened to one of their track at least once
Plot the results, and explain the figures you obtain.

# calculate top-10 artists in term of play counts
top10ArtistsPC = newUserArtistDF.groupBy('artistID').sum('playCount').orderBy('sum(playCount)', ascending=0).take(10)

y_pos = list(range(len(top10ArtistsPC)))
pdf = pd.DataFrame(data=top10ArtistsPC)

plt.barh(y_pos, pdf[1][::-1], align='center', alpha=0.4)
plt.yticks(y_pos, pdf[0][::-1])
plt.xlabel('Play Count')
plt.ylabel('Artist')
plt.title('Top-10 Artist ID per play counts')
plt.show()

After removing all mispelledID, there is little increase in playCounts of top-10-artists.
Here I show the comparison on playCount of top 10 artist before and after removing mispelledID
artistID=979: 2502130 -> 2502596.
artistID=1000113: 2259185 -> 2259825.
artistID=4267: 1930592 -> 1931143.
artistID=1000024: 1542806 -> 1543430.
artistID=4468: 1425942 -> 1426254.
artistID=82: 1399418 -> 1399665.
artistID=831: 1361392 -> 1361977.
artistID=1001779: 1328869 -> 1328969.
artistID=1000130: 1234387 -> 1234773.
artistID=976: 1203226 -> 1203348.
The chart shows that playCount dramatiaclly decreases from the 1st artist to the 4th artist in top-10-artists, but from the 4th artist to 10th artist, it decreases slightly.
Who are the top-10 users?
In terms of absolute play counts
In terms of "curiosity", that is, how many different artists they listened to
Plot the results

# calculate top 10 users in term of play counts
top10UsersByPlayCount = newUserArtistDF.groupBy("userID").sum('playCount').orderBy('sum(playCount)', ascending=0).take(10)

y_pos = list(range(len(top10UsersByPlayCount)))
pdf = pd.DataFrame(data=top10UsersByPlayCount)

plt.barh(y_pos, pdf[1][::-1], align='center', alpha=0.4)
plt.yticks(y_pos, pdf[0][::-1])
plt.xlabel('Play Count') 
plt.ylabel('User')
plt.title('Top-10 Users ID per play counts')
plt.show()

top10UsersByCuriosity = (newUserArtistDF.dropDuplicates(['userID', 'artistID'])
                             .groupBy("userID")
                             .count()
                             .orderBy('count', ascending=0)
                             .take(10)
                         )

#print(top10UsersByCuriosity)
y_pos = range(len(top10UsersByCuriosity))

pdf = pd.DataFrame(data=top10UsersByCuriosity)

plt.barh(y_pos, pdf[1][::-1], align='center', alpha=0.4)
plt.yticks(y_pos, pdf[0][::-1])
plt.xlabel('Number of played artists')
plt.ylabel('User')
plt.title('Top-10 Users ID per Curiosity')
plt.show()

Now we have some valuable information about the data. It's the time to study how to build a statistical models.

2. Build a statistical models to make recommendations
2.1 Introduction to recommender systems
In a recommendation-system application there are two classes of entities, which we shall refer to as users and items. Users have preferences for certain items, and these preferences must be inferred from the data. The data itself is represented as a preference matrix , giving for each user-item pair, a value that represents what is known about the degree of preference of that user for that item. The table below is an example for a preference matrix of 5 users and k items. The preference matrix is also known as utility matrix.

\	IT1	IT2	IT3	...	ITk
U1	1	...	5	...	3
U2	...	2	...	...	2
U3	5	...	3	...	...
U4	3	3	...	...	4
U5	...	1	...	...	...
The value of row i, column j expresses how much does user i like item j. The values are often the rating scores of users for items. An unknown value implies that we have no explicit information about the user's preference for the item. The goal of a recommendation system is to predict "the blanks" in the preference matrix. For example, assume that the rating score is from 1 (dislike) to 5 (love), would user U5 like IT3 ? We have two approaches:

Designing our recommendation system to take into account properties of items such as brand, category, price... or even the similarity of their names. We can denote the similarity of items IT2 and IT3, and then conclude that because user U5 did not like IT2, they were unlikely to enjoy SW2 either.

We might observe that the people who rated both IT2 and IT3 tended to give them similar ratings. Thus, we could conclude that user U5 would also give IT3 a low rating, similar to U5's rating of IT2

It is not necessary to predict every blank entry in a utility matrix. Rather, it is only necessary to discover some entries in each row that are likely to be high. In most applications, the recommendation system does not oﬀer users a ranking of all items, but rather suggests a few that the user should value highly. It may not even be necessary to ﬁnd all items with the highest expected ratings, but only to ﬁnd a large subset of those with the highest ratings.

2.2 Families of recommender systems
In general, recommender systems can be categorized into two groups:

Content-Based systems focus on properties of items. Similarity of items is determined by measuring the similarity in their properties.

Collaborative-Filtering systems focus on the relationship between users and items. Similarity of items is determined by the similarity of the ratings of those items by the users who have rated both items.

In the usecase of this notebook, artists take the role of items, and users keep the same role as users. Since we have no information about artists, except their names, we cannot build a content-based recommender system.

Therefore, in the rest of this notebook, we only focus on Collaborative-Filtering algorithms.

2.3 Collaborative-Filtering
In this section, we study a member of a broad class of algorithms called latent-factor models. They try to explain observed interactions between large numbers of users and products through a relatively small number of unobserved, underlying reasons. It is analogous to explaining why millions of people buy a particular few of thousands of possible albums by describing users and albums in terms of tastes for perhaps tens of genres, tastes which are not directly observable or given as data.

First, we formulate the learning problem as a matrix completion problem. Then, we will use a type of matrix factorization model to "fill in" the blanks. We are given implicit ratings that users have given certain items (that is, the number of times they played a particular artist) and our goal is to predict their ratings for the rest of the items. Formally, if there are  users and  items, we are given an  matrix  in which the generic entry  represents the rating for item  by user . Matrix  has many missing entries indicating unobserved ratings, and our task is to estimate these unobserved ratings.

A popular approach to the matrix completion problem is matrix factorization, where we want to "summarize" users and items with their latent factors.

Parallel Altenating Least Squares using broadcast variables
The approach takes advantage of the fact that the  and  factor matrices are often very small and can be stored locally on each machine.

Partition the Ratings RDD by user to create , and similarly partition the Ratings RDD by item to create . This means there are two copies of the same Ratings RDD, albeit with different partitionings. In , all ratings by the same user are on the same machine, and in  all ratings for same item are on the same machine.
Broadcast the matrices  and . Note that these matrices are not RDD of vectors: they are now "local: matrices.
Using  and , we can use expression  from above to compute the update of  locally on each machine
Using  and , we can use expression  from above to compute the update of  locally on each machine
A further optimization to this method is to group the  and  factors matrices into blocks (user blocks and item blocks) and reduce the communication by only sending to each machine the block of users (or items) that are needed to compute the updates at that machine.

This method is called Block ALS. It is achieved by precomputing some information about the ratings matrix to determine the "out-links" of each user (which blocks of the items it will contribute to) and "in-link" information for each item (which of the factor vectors it receives from each user block it will depend on). For exmple, assume that machine 1 is responsible for users 1,2,...,37: these will be block 1 of users. The items rated by these users are block 1 of items. Only the factors of block 1 of users and block 1 of items will be broadcasted to machine 1.

Further readings
Other methods for matrix factorization include:

Low Rank Approximation and Regression in Input Sparsity Time, by Kenneth L. Clarkson, David P. Woodruff. http://arxiv.org/abs/1207.6365
Generalized Low Rank Models (GLRM), by Madeleine Udell, Corinne Horn, Reza Zadeh, Stephen Boyd. http://arxiv.org/abs/1410.0342
Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares, by Trevor Hastie, Rahul Mazumder, Jason D. Lee, Reza Zadeh . Statistics Department and ICME, Stanford University, 2014. http://stanford.edu/~rezab/papers/fastals.pdf
3. Usecase : Music recommender system
In this usecase, we use the data of users and artists in the previous sections to build a statistical model to recommend artists for users.

3.1 Requirements
According to the properties of data, we need to choose a recommender algorithm that is suitable for this implicit feedback data. It means that the algorithm should learn without access to user or artist attributes such as age, genre,.... Therefore, an algorithm of type collaborative filtering is the best choice.

Second, in the data, there are some users that have listened to only 1 artist. We need an algorithm that might provide decent recommendations to even these users. After all, at some point, every user starts out with just one play at some point!

Third, we need an algorithm that scales, both in its ability to build large models, and to create recommendations quickly. So, an algorithm which can run on a distributed system (SPARK, Hadoop...) is very suitable.

From these requirement, we can choose using ALS algorithm in SPARK's MLLIB.

Spark MLlib’s ALS implementation draws on ideas from 1 and 2.

3.2 Notes
Currently, MLLIB can only build models from an RDD. That means we have two ways to prepare data:

Loading to into SPARK SQL DataFrame as before, and then access the corresponding RDD by calling <dataframe>.rdd. The invalid data is often sucessfully dropped by using mode DROPMALFORMED. However, this way might not work in all cases. Fortunately, we can use it with this usecase.

Loading data directly to RDD. However, we have to deal with the invalid data ourself. In the trade-off, this way is the most reliable, and can work in every case.

In this notebook, we will use the second approach: it requires a bit more effort, but the reward is worth it!

3.3 Cleanup the data
In section Data Integration, I already replaced the ids of mispelled artist IDs by the corresponding standard ids by using SPARK SQL API. However, if the data has the invalid entries such that SPARK SQL API is stuck, the best way to work with it is using an RDD.

Just as a recall, I work with three datasets in user_artist_data.txt, ` andartist_alias.txt`. The entries in these file can be empty or have only one field.

In details our goal now is:

Read the input user_artist_data.txt and transforms its representation into an output dataset.
To produce an output "tuple" containing the original user identifier and play counts, but with the artist identifier replaced by its most common alias, as found in the artist_alias.txt dataset.
Since the artist_alias.txt file is small, we can use a technique called broadcast variables to make such transformation more efficient.
Load data from /dataset/artist_alias.txt and filter out the invalid entries to construct a dictionary to map from mispelled artists' ids to standard ids.
NOTE:

From now on, we will use the "standard" data to train our model.

If a line contains less than 2 fields or contains invalid numerial values, we can return a special tuple. After that, we can filter out these special tuples.

rawArtistAlias = sc.textFile(base + "artist_alias.txt")

def xtractFields(s):
    # Using white space or tab character as separetors,
    # split a line into list of strings 
    line = re.split("\s|\t",s,1)
    # if this line has at least 2 characters
    if (len(line) > 1):
        try:
            # try to parse the first and the second components to integer type
            return (int(line[0]), int(line[1]))
        except ValueError:
            # if parsing has any error, return a special tuple
            return (-1,-1)
    else:
        # if this line has less than 2 characters, return a special tuple
        return (-1,-1)

artistAlias = (
                rawArtistAlias
                    # extract fields using function xtractFields
                    .map( lambda x: xtractFields(x))

                    # fileter out the special tuples
                    .filter( lambda x: x != (-1,-1) )
                    # collect result to the driver as a "dictionary"
                    .collectAsMap()
                )
Prepare RDD userArtistDataRDD by replacing mispelled artists' ids to standard ids. Show 5 samples.
Using broadcast varible can help us increase the effiency.

bArtistAlias = sc.broadcast(artistAlias)
rawUserArtistData = sc.textFile(base + "user_artist_data.txt")

def disambiguate(line):
    [userID, artistID, count] = line.split(' ')
    finalArtistID = bArtistAlias.value.get(artistID,artistID)
    return (userID, finalArtistID,count)


userArtistDataRDD = rawUserArtistData.map(disambiguate)
userArtistDataRDD.take(5)
[('1000002', '1', '55'),
 ('1000002', '1000006', '33'),
 ('1000002', '1000007', '8'),
 ('1000002', '1000009', '144'),
 ('1000002', '1000010', '314')]
3.4 Training our statistical model
To train a model using ALS, we must use a preference matrix as an input. MLLIB uses the class Rating to support the construction of a distributed preference matrix.

Given RDD userArtistDataRDD in previous section, construct a new RDD called trainingData by tranforming each item of it into a Rating object.
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
allData = userArtistDataRDD.map(lambda r: Rating( float(r[0]),float(r[1]),int(r[2]))) \
                                            .repartition(12).cache()
allData.take(5)
[Rating(user=1038228, product=1027389, rating=1.0),
 Rating(user=1038228, product=10280656, rating=4.0),
 Rating(user=1038228, product=1028633, rating=2.0),
 Rating(user=1038228, product=1028971, rating=1.0),
 Rating(user=1038228, product=1029494, rating=3.0)]
A model can be trained by using ALS.trainImplicit(<training data>, <rank>), where:
training data is the input data you decide to feed to the ALS algorithm
rank is the number of laten features
We can also use some additional parameters to adjust the quality of the model. Currently, let's set

rank=10
iterations=5
lambda_=0.01
alpha=1.0
to build model.

#setting parameters
rank=10
iterations=5
lambda_=0.01
alpha=1.0

#training
t0 = time()
model = ALS.trainImplicit(allData, rank)
t1 = time()
print("finish training model in %f secs" % (t1 - t0))
finish training model in 91.764442 secs
The trained model can be saved into HDFS for later use. This can be done via model.save(sc, <file_name>).
Let's use this function to store our model as name music_rec_model.spark.

#! hdfs dfs -rm -R -f -skipTrash lastfm_model.spark
#print('Delete old model')
model.save(sc,'music_rec_model.spark')
print('Save new model')
Save new model
A saved model can be load from file by using MatrixFactorizationModel.load(sc, <file_name>).
Let's load the saved model from file.

t0 = time()
model = MatrixFactorizationModel.load(sc, 'music_rec_model.spark')
t1 = time()
print("finish loading model in %f secs" % (t1 - t0))
finish loading model in 1.624458 secs
Print the first row of user features in our model.
model.userFeatures().first()
(90,
 array('d', [-0.05403498187661171, -0.00984848104417324, -0.0005837124772369862, -0.04152700677514076, 0.003350968472659588, -0.06338433176279068, -0.035387467592954636, 0.07213515043258667, -0.0294994805008173, -0.03624201565980911]))
Show the top-5 artist names recommendated for user 2093760.
The recommendations can be given by function recommendProducts(userID, num_recommendations). These recommendations are only artist ids. We have to map them to artist names by using data in artist_data.txt.

# Make five reccommendations to user 2093760
recommendations = (model.recommendProducts(2093760,5))
print(recommendations)
# construct set of recommendated artists
recArtist = set(rating[1] for rating in recommendations)
recArtist
[Rating(user=2093760, product=1007614, rating=0.030772450232132396), Rating(user=2093760, product=4605, rating=0.029059107013603248), Rating(user=2093760, product=2814, rating=0.02767556526645912), Rating(user=2093760, product=829, rating=0.0275336505441955), Rating(user=2093760, product=1037970, rating=0.026948925288733955)]
{829, 2814, 4605, 1007614, 1037970}
# construct data of artists (artist_id, artist_name)

rawArtistData = sc.textFile(base + "artist_data.txt")

def xtractFields(s):
    line = re.split("\s|\t",s,1)
    if (len(line) > 1):
        try:
            return (int(line[0]), str(line[1].strip()))
        except ValueError:
            return (-1,"")
    else: 
        return (-1,"")

artistByID = rawArtistData.map(xtractFields).filter(lambda x: x[0] > 0)
# Filter in those artists, get just artist, and print
def artistNames(line):
#     [artistID, name]
    if (line[0] in recArtist):
        return True
    else:
        return False

recList = artistByID.filter(artistNames).values().collect()

print(recList)
['50 Cent', 'Snoop Dogg', 'Nas', 'Jay-Z', 'Kanye West']
IMPORTANT NOTE
At the moment, it is necessary to manually unpersist the RDDs inside the model when you are done with it. The following function can be used to make sure models are promptly uncached.

def unpersist(model):
    model.userFeatures().unpersist()
    model.productFeatures().unpersist()

# uncache data and model when they are no longer used  
unpersist(model)
3.5 Evaluating Recommendation Quality
In this section, we study how to evaluate the quality of our model. It's hard to say how good the recommendations are. One of serveral methods approach to evaluate a recommender based on its ability to rank good items (artists) high in a list of recommendations. The problem is how to define "good artists". Currently, by training all data, "good artists" is defined as "artists the user has listened to", and the recommender system has already received all of this information as input. It could trivially return the users previously-listened artists as top recommendations and score perfectly. Indeed, this is not useful, because the recommender's is used to recommend artists that the user has never listened to.

To overcome that problem, we can hide the some of the artist play data and only use the rest to train model. Then, this held-out data can be interpreted as a collection of "good" recommendations for each user. The recommender is asked to rank all items in the model, and the rank of the held-out artists are examined. Ideally the recommender places all of them at or near the top of the list.

The recommender's score can then be computed by comparing all held-out artists' ranks to the rest. The fraction of pairs where the held-out artist is ranked higher is its score. 1.0 is perfect, 0.0 is the worst possible score, and 0.5 is the expected value achieved from randomly ranking artists.

AUC(Area Under the Curve) can be used as a metric to evaluate model. It is also viewed as the probability that a randomly-chosen "good" artist ranks above a randomly-chosen "bad" artist.

Next, we split the training data into 2 parts: trainData and cvData with ratio 0.7:0.3 respectively, where trainData is the dataset that will be used to train model. Then we write a function to calculate AUC to evaluate the quality of our model.

Split the data into trainData and cvData with ratio 0.9:0.1 and use the first part to train a statistic model with:
rank=10
iterations=5
lambda_=0.01
alpha=1.0
trainData, cvData = allData.randomSplit([0.7,0.3],1)
trainData.cache()
cvData.cache()
PythonRDD[332] at RDD at PythonRDD.scala:49
Here I just split the dataset to 70% for training and 30% for testing. If I have enough time, I will conduct an experiment that changing the proportion of splitting data to see how it affects our prediction model. (See it at Addition work)
#training
t0 = time()
model = ALS.trainImplicit(ratings=trainData,rank=rank,iterations=iterations,lambda_=lambda_ ,alpha=alpha)
t1 = time()
print("finish training model in %f secs" % (t1 - t0))
finish training model in 74.359045 secs
Area under the ROC curve: a function to compute it
# Get all unique artistId, and broadcast them
allItemIDs = np.array(allData.map(lambda x: x[1]).distinct().collect())
bAllItemIDs = sc.broadcast(allItemIDs)
from random import randint

# Depend on the number of item in userIDAndPosItemIDs,
# create a set of "negative" products for each user. These are randomly chosen
# from among all of the other items, excluding those that are "positive" for the user.
# NOTE 1: mapPartitions operates on many (user,positive-items) pairs at once
# NOTE 2: flatMap breaks the collections above down into one big set of tuples
def xtractNegative(userIDAndPosItemIDs):
    def pickEnoughNegatives(line):
        userID = line[0]
        posItemIDSet = set(line[1])
        #posItemIDSet = line[1]
        negative = []
        allItemIDs = bAllItemIDs.value
        # Keep about as many negative examples per user as positive. Duplicates are OK.
        i = 0
        while (i < len(allItemIDs) and len(negative) < len(posItemIDSet)):
            itemID = allItemIDs[randint(0,len(allItemIDs)-1)]
            if itemID not in posItemIDSet:
                negative.append(itemID)
            i += 1
        
        # Result is a collection of (user,negative-item) tuples
        return map(lambda itemID: (userID, itemID), negative)

    # Init an RNG and the item IDs set once for partition
    # allItemIDs = bAllItemIDs.value
    return map(pickEnoughNegatives, userIDAndPosItemIDs)

def ratioOfCorrectRanks(positiveRatings, negativeRatings):
    
    # find number elements in arr that has index >= start and has value smaller than x
    # arr is a sorted array
    def findNumElementsSmallerThan(arr, x, start=0):
        left = start
        right = len(arr) -1
        # if x is bigger than the biggest element in arr
        if start > right or x > arr[right]:
            return right + 1
        mid = -1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] < x:
                left = mid + 1
            elif arr[mid] > x:
                right = mid - 1
            else:
                while mid-1 >= start and arr[mid-1] == x:
                    mid -= 1
                return mid
        return mid if arr[mid] > x else mid + 1
    
    ## AUC may be viewed as the probability that a random positive item scores
    ## higher than a random negative one. Here the proportion of all positive-negative
    ## pairs that are correctly ranked is computed. The result is equal to the AUC metric.
    correct = 0 ## L
    total = 0 ## L
    
    # sorting positiveRatings array needs more cost
    #positiveRatings = np.array(map(lambda x: x.rating, positiveRatings))

    negativeRatings = list(map(lambda x:x.rating, negativeRatings))
    
    #np.sort(positiveRatings)
    negativeRatings.sort()# = np.sort(negativeRatings)
    total = len(positiveRatings)*len(negativeRatings)
    
    for positive in positiveRatings:
        # Count the correctly-ranked pairs
        correct += findNumElementsSmallerThan(negativeRatings, positive.rating)
        
    ## Return AUC: fraction of pairs ranked correctly
    return float(correct) / total

def calculateAUC(positiveData, bAllItemIDs, predictFunction):
    # Take held-out data as the "positive", and map to tuples
    positiveUserProducts = positiveData.map(lambda r: (r[0], r[1]))
    # Make predictions for each of them, including a numeric score, and gather by user
    positivePredictions = predictFunction(positiveUserProducts).groupBy(lambda r: r.user)
    
    # Create a set of "negative" products for each user. These are randomly chosen 
    # from among all of the other items, excluding those that are "positive" for the user. 
    negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions(xtractNegative).flatMap(lambda x: x)
    # Make predictions on the rest
    negativePredictions = predictFunction(negativeUserProducts).groupBy(lambda r: r.user)
    
    return (
            positivePredictions.join(negativePredictions)
                .values()
                .map(
                    lambda positive_negativeRatings: ratioOfCorrectRanks(positive_negativeRatings[0], positive_negativeRatings[1])
                )
                .mean()
            )
Using part cvData and function calculateAUC to compute the AUC of the trained model.
t0 = time()
auc = calculateAUC( cvData,bAllItemIDs, model.predictAll)
t1 = time()
print("auc=",auc)
print("finish in %f seconds" % (t1 - t0))
auc= 0.96070668941573
finish in 79.103230 seconds
Now we have the AUC of our model, it’s helpful to benchmark this against a simpler approach. For example, consider recommending the globally most-played artists to every user. This is not personalized, but is simple and may be effective.
Implement this simple pupolarity-based prediction algorithm, evaluate its AUC score, and compare to the results achieved by the more sophisticated ALS algorithm.

bListenCount = sc.broadcast(trainData.map(lambda r: (r[1], r[2])).reduceByKey(lambda x,y: x+y).collectAsMap())
def predictMostListened(allData):
    return allData.map(lambda r: Rating(r[0], r[1], bListenCount.value.get( r[1], 0.0)))
auc = calculateAUC(cvData,bListenCount, predictMostListened)
print("AUC score:" + str(auc))
AUC score:0.9361065418733973
3.6 Personalized recommendations with ALS: Hyperparameters tuning
In the previous section, we build our models with some given paramters without any knowledge about them. Actually, choosing the best parameters' values is very important. It can significantly affect the quality of models. Especially, with the current implementation of ALS in MLLIB, these parameters are not learned by the algorithm, and must be chosen by the caller. The following parameters should get consideration before training models:

rank = 10: the number of latent factors in the model, or equivalently, the number of columns  in the user-feature and product-feature matrices. In non-trivial cases, this is also their rank.

iterations = 5: the number of iterations that the factorization runs. Instead of runing the algorithm until RMSE converged which actually takes very long time to finish with large datasets, we only let it run in a given number of iterations. More iterations take more time but may produce a better factorization.

lambda_ = 0.01: a standard overfitting parameter. Higher values resist overfitting, but values that are too high hurt the factorization's accuracy.

alpha = 1.0: controls the relative weight of observed versus unobserved userproduct interactions in the factorization.

Although all of them have impact on the models' quality, iterations is more of a constraint on resources used in the factorization. So, rank, lambda_ and alpha can be considered hyperparameters to the model. We will try to find "good" values for them. Indeed, the values of hyperparameter are not necessarily optimal. Choosing good hyperparameter values is a common problem in machine learning. The most basic way to choose values is to simply try combinations of values and evaluate a metric for each of them, and choose the combination that produces the best value of the metric.

Grid Search
For simplicity, assume that we want to explore the following parameter space: ,  and .

Find the best combination of them in terms of the highest AUC value.

evaluations = []

for rank in [10, 50]:
    for lambda_ in [1.0, 0.0001]:
        for alpha in [1.0, 40.0]:
            print("Train model with rank=%d lambda_=%f alpha=%f" % (rank, lambda_, alpha))
            # with each combination of params, we should run multiple times and get avg
            # for simple, we only run one time.
            model = ALS.trainImplicit(ratings=trainData,rank=rank,iterations=5,lambda_=lambda_ ,alpha=alpha)
            auc = calculateAUC(cvData,bListenCount,model.predictAll)

            evaluations.append(((rank, lambda_, alpha), auc))

            unpersist(model)

evaluations.sort(key = lambda x: -x[1])
evalDataFrame = pd.DataFrame(data=evaluations)
print(evalDataFrame)

trainData.unpersist()
cvData.unpersist()
Train model with rank=10 lambda_=1.000000 alpha=1.000000
Train model with rank=10 lambda_=1.000000 alpha=40.000000
Train model with rank=10 lambda_=0.000100 alpha=1.000000
Train model with rank=10 lambda_=0.000100 alpha=40.000000
Train model with rank=50 lambda_=1.000000 alpha=1.000000
Train model with rank=50 lambda_=1.000000 alpha=40.000000
Train model with rank=50 lambda_=0.000100 alpha=1.000000
Train model with rank=50 lambda_=0.000100 alpha=40.000000
                    0         1
0     (10, 1.0, 40.0)  0.973829
1  (10, 0.0001, 40.0)  0.971861
2     (50, 1.0, 40.0)  0.971527
3  (50, 0.0001, 40.0)  0.970065
4      (10, 1.0, 1.0)  0.964493
5      (50, 1.0, 1.0)  0.959280
6   (10, 0.0001, 1.0)  0.958409
7   (50, 0.0001, 1.0)  0.942701
PythonRDD[349] at RDD at PythonRDD.scala:49
The combination of parameters that gets the highest AUC is: rank = 10 ; lambda = 1.0 ; alpha = 40
Using "optimal" hyper-parameters obtained, re-train the model and show top-5 artist names recommendated for user 2093760.
model = ALS.trainImplicit(ratings=trainData, rank=10, iterations=5, lambda_=1.0, alpha=40.0)
allData.unpersist()

userID = 2093760
recommendations = model.recommendProducts(userID,5)

recArtist = set(rating[1] for rating in recommendations)

# Filter in those artists, get just artist, and print
def artistNames(line):
#     [artistID, name]
    if (line[0] in recArtist):
        return True
    else:
        return False

recList = artistByID.filter(artistNames).values().collect()
print(recList)

unpersist(model)
['Kent', 'Oasis', 'The Killers', 'Kaiser Chiefs', 'Unknown']
It seems that the result of top 5 recommended artists does not change when we add more parameters to the model or modify them (such as lambda, alpha). I think that we should extend to top 50 or top 70 to see how the recommendation changing. Because I think that lambda and alpha parameters affect only when the rating of each artist approximate the mean of total ratings for one user (Top 5 artists have the much larger rating value than others, top 5 artists do not change with modified parameter)
This raises me the question that how strong the impact of those parameters? Is it really nessessary to put into our model when we just need to retrieve a small number of top artists (let's say, less than top 10)?
Additional work
1/ Changing the proportion of splitting data
In this experiment, I will change the percentage of training data to 50%, 80%, 90% and 99% respectively. The purpose is to observe the changing of AUC score
trainData, cvData = allData.randomSplit([0.5,0.5],1)
trainData.cache()
cvData.cache()

t0 = time()
model = ALS.trainImplicit(ratings=trainData,rank=rank,iterations=iterations,lambda_=lambda_ ,alpha=alpha)
t1 = time()
print("finish training model in %f secs" % (t1 - t0))

t0 = time()
auc = calculateAUC( cvData,bAllItemIDs, model.predictAll)
t1 = time()
print("auc=",auc)
print("finish in %f seconds" % (t1 - t0))
finish training model in 242.592275 secs
auc= 0.9721443801890255
finish in 130.203251 seconds
trainData, cvData = allData.randomSplit([0.8,0.2],1)
trainData.cache()
cvData.cache()

t0 = time()
model = ALS.trainImplicit(ratings=trainData,rank=rank,iterations=iterations,lambda_=lambda_ ,alpha=alpha)
t1 = time()
print("finish training model in %f secs" % (t1 - t0))

t0 = time()
auc = calculateAUC( cvData,bAllItemIDs, model.predictAll)
t1 = time()
print("auc=",auc)
print("finish in %f seconds" % (t1 - t0))
finish training model in 338.598379 secs
auc= 0.9808745215544642
finish in 61.614985 seconds
trainData, cvData = allData.randomSplit([0.9,0.1],1)
trainData.cache()
cvData.cache()

t0 = time()
model = ALS.trainImplicit(ratings=trainData,rank=rank,iterations=iterations,lambda_=lambda_ ,alpha=alpha)
t1 = time()
print("finish training model in %f secs" % (t1 - t0))

t0 = time()
auc = calculateAUC( cvData,bAllItemIDs, model.predictAll)
t1 = time()
print("auc=",auc)
print("finish in %f seconds" % (t1 - t0))
finish training model in 247.797857 secs
auc= 0.9847853981146301
finish in 39.068620 seconds
trainData, cvData = allData.randomSplit([0.99,0.01],1)
trainData.cache()
cvData.cache()

t0 = time()
model = ALS.trainImplicit(ratings=trainData,rank=rank,iterations=iterations,lambda_=lambda_ ,alpha=alpha)
t1 = time()
print("finish training model in %f secs" % (t1 - t0))

t0 = time()
auc = calculateAUC( cvData,bAllItemIDs, model.predictAll)
t1 = time()
print("auc=",auc)
print("finish in %f seconds" % (t1 - t0))
finish training model in 325.923175 secs
auc= 0.9856858729938448
finish in 16.474116 seconds
As the above results, we have:
training 50% -> AUC: 0.9557431751634355
training 80% -> AUC: 0.9633452139683835
training 90% -> AUC: 0.9641551756487843
training 99% -> AUC: 0.9672845649647493
This is obviouly that the proportion of splitting data does affect the result of the AUC score. The more training data, the higher AUC score. In my opinion, having higher AUC score does not mean that the model is working well in general. Because high percentage of training data may prone to overfitting.
2/ Expand the the list of recommended artist
As my hypothesis, in this experiment, instead of retrieving top 5 artist, I try to retrieve top 50 and 70 artist with modified parameters to see the differences the result compare to the model with standard parameters
Model with standard parameters (lambda and alpha are set by default)
trainData, cvData = allData.randomSplit([0.7,0.3],1)
trainData.cache()
cvData.cache()
PythonRDD[3102] at RDD at PythonRDD.scala:49
model50 = ALS.trainImplicit(ratings=trainData, rank=10, iterations=5)
trainData.unpersist()

userID = 2093760
recommendations50 = model50.recommendProducts(userID,50)

recArtist = set(rating[1] for rating in recommendations50)

recList50 = artistByID.filter(artistNames).values().collect()

print(recList50)

unpersist(model)
['Eric Clapton', 'Notorious B.I.G.', '50 Cent', 'Sublime', 'Snoop Dogg', 'RJD2', 'The Postal Service', 'De La Soul', 'Nas', 'Jay-Z', 'Zero 7', 'The Chemical Brothers', 'Kanye West', 'N*E*R*D', 'Wu-Tang Clan', 'Daft Punk', 'Mos Def', 'Cake', 'Bob Marley', 'Dr. Dre', 'Ludacris', 'Rage Against the Machine', '2Pac', 'Jack Johnson', 'The Beatles', 'Eminem', 'Modest Mouse', 'Pearl Jam', 'Thievery Corporation', 'A Tribe Called Quest', 'Jimi Hendrix', 'Led Zeppelin', 'Radiohead', 'Red Hot Chili Peppers', 'Coldplay', 'Dave Matthews Band', 'U2', 'Gorillaz', 'Incubus', 'Miles Davis', 'DJ Shadow', 'Pink Floyd', 'Jurassic 5', 'Beck', 'Outkast', 'The Roots', 'The Game', 'Talib Kweli', '311', 'Beastie Boys']
Model with lamda = 0.0001 , alpha = 40
model50_1 = ALS.trainImplicit(ratings=trainData, rank=10, iterations=5, lambda_= 0.0001, alpha= 40.0)
trainData.unpersist()

userID = 2093760
recommendations50_1 = model50_1.recommendProducts(userID,50)

recArtist = set(rating[1] for rating in recommendations50_1)

recList50_1 = artistByID.filter(artistNames).values().collect()

print(recList50_1)

unpersist(model)
['Bloc Party', 'Snow Patrol', '[unknown]', 'The Postal Service', 'Death Cab for Cutie', 'Keane', 'Damien Rice', 'Bob Dylan', 'Johnny Cash', 'Elliott Smith', 'The Who', 'Cake', 'Morrissey', 'Bruce Springsteen', 'The Shins', 'Jack Johnson', 'Jeff Buckley', 'Bright Eyes', 'The Clash', 'Pixies', 'The Smiths', 'The Beatles', 'Modest Mouse', 'The Strokes', 'The White Stripes', 'R.E.M.', 'Franz Ferdinand', 'Simon & Garfunkel', 'The Cure', 'Led Zeppelin', 'Radiohead', 'David Bowie', 'The Mars Volta', 'The Killers', 'Moby', 'Coldplay', 'U2', 'Blur', 'The Beach Boys', 'Weezer', 'Interpol', 'Pink Floyd', 'The Rolling Stones', 'Beck', 'Kaiser Chiefs', 'They Might Be Giants', 'Ben Folds', 'Beastie Boys', 'Björk', 'Hot Hot Heat']
temp3 = [item for item in recList50 if item not in recList50_1]
print(temp3)
print("\n Number of different artists between standard model and model_1:" + str(len(temp3)))
['Eric Clapton', 'Notorious B.I.G.', '50 Cent', 'Sublime', 'Snoop Dogg', 'RJD2', 'De La Soul', 'Nas', 'Jay-Z', 'Zero 7', 'The Chemical Brothers', 'Kanye West', 'N*E*R*D', 'Wu-Tang Clan', 'Daft Punk', 'Mos Def', 'Bob Marley', 'Dr. Dre', 'Ludacris', 'Rage Against the Machine', '2Pac', 'Eminem', 'Pearl Jam', 'Thievery Corporation', 'A Tribe Called Quest', 'Jimi Hendrix', 'Red Hot Chili Peppers', 'Dave Matthews Band', 'Gorillaz', 'Incubus', 'Miles Davis', 'DJ Shadow', 'Jurassic 5', 'Outkast', 'The Roots', 'The Game', 'Talib Kweli', '311']

 Number of different artists between standard model and model_1:38
Model with lambda = 0.1, alpha = 1.0
model50_2 = ALS.trainImplicit(ratings=trainData, rank=10, iterations=5, lambda_= 0.1, alpha= 1.0)
trainData.unpersist()

userID = 2093760
recommendations50_2 = model50_2.recommendProducts(userID,50)

recArtist = set(rating[1] for rating in recommendations50_2)

recList50_2 = artistByID.filter(artistNames).values().collect()

print(recList50_2)

unpersist(model)
['Notorious B.I.G.', 'Fabolous', '50 Cent', 'Mobb Deep', 'DMX', 'Snoop Dogg', 'Usher', 'Nelly', 'Busta Rhymes', 'De La Soul', 'Nas', 'Ja Rule', 'Jay-Z', 'Will Smith', 'Black Eyed Peas', 'Gang Starr', 'Kanye West', 'N*E*R*D', 'D12', 'Wu-Tang Clan', 'Mos Def', 'Bob Marley', 'Dr. Dre', 'Ludacris', '2Pac', 'Eminem', 'A Tribe Called Quest', 'Bob Marley & the Wailers', 'Jurassic 5', 'Jedi Mind Tricks', 'Method Man', 'Outkast', 'The Roots', 'Lil Jon & The East Side Boyz', 'Xzibit', 'G-Unit', 'The Game', 'Lloyd Banks', 'Jay-Z and Linkin Park', 'Bone Thugs-N-Harmony', 'Talib Kweli', 'Obie Trice', 'Chingy', 'Atmosphere', 'Dilated Peoples', 'N.W.A', 'Twista', 'Cypress Hill', 'Ice Cube', 'Common']
temp4 = [item for item in recList50 if item not in recList50_2]
print(temp4)
print("\n Number of different artists between standard model and model_2:" + str(len(temp4)))
['Eric Clapton', 'Sublime', 'RJD2', 'The Postal Service', 'Zero 7', 'The Chemical Brothers', 'Daft Punk', 'Cake', 'Rage Against the Machine', 'Jack Johnson', 'The Beatles', 'Modest Mouse', 'Pearl Jam', 'Thievery Corporation', 'Jimi Hendrix', 'Led Zeppelin', 'Radiohead', 'Red Hot Chili Peppers', 'Coldplay', 'Dave Matthews Band', 'U2', 'Gorillaz', 'Incubus', 'Miles Davis', 'DJ Shadow', 'Pink Floyd', 'Beck', '311', 'Beastie Boys']

 Number of different artists between standard model and model_2:29
Conclusion: As the result has shown above, I dont need to expand the result list to top 70 artists anymore. It is clearly proving my hypothesis is true
How we configure the lambda and alpha makes a huge impact on the retrieved result: In my example, the top artists that highly recommended to user="2093760" is ['50Cent', 'Snoopdog', Notorious B.I.G] (they appear in both standard model and modified models). Others artists may vary from different models.
We may not need to include lambda and alpha to the model if we only retreive the top result less than 10 artists.
Proposed method:
In my opinion: We should cut out outliers (top 5 artists). With this method, recommendation will be more diverse and accurate.
Summary
In this notebook, we introduce an algorithm to do matrix factorization and the way of using it to make recommendation. Further more, we studied how to build a large-scale recommender system on SPARK using ALS algorithm and evaluate its quality. Finally, a simple approach to choose good parameters is mentioned.
