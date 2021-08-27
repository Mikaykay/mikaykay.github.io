---
layout: post
title: Predicting Credit Card Fraud
subtitle: Using Classification models on a data set containing legitimate and fraudulent transactions.
gh-repo: mikaykay/
gh-badge: [star, fork, follow]
tags: [Regression,Tree Models, Finance]
comments: true
---

About a few years ago I had some fraudulent transactions made from my credit card and when I first saw them it made me confused and frustrated. I was questioning if I really made those transactions or if I recently signed up for a new subscription to something. I had to go back into my emails and look to see what I recently bought to double check. In the end, I called my bank asking about more information about the transaction and we both deemed it to be fraud. The bank took care of everything, but it was scary to think that someone was using my card to make payments and if I didn’t create the habit to go on my banking app about every week then the situation could have been worse. 


In the moment the whole situation made me feel like I couldn’t trust the stores and gas stations with making a simple transaction without being paranoid about my credit card information being stolen. Ever since then I developed a habit of going on my banking app almost every night before going to bed, making sure I cover my card keypad while entering any PIN and paying with cash at gas stations. 


Naturally a credit card fraud data set caught my eye when looking for a data set to apply a predictive model to.  


# The Data Set
Looking for a data set was extremely difficult, all the data sets I wanted to try were related to more advanced techniques like AI, Object Recognition and NLP methods. It was difficult for me to curb my ambition and find something simpler for practicing prediction models on. 
Eventually I stumbled on a Kaggle set about detecting credit card fraud
## Before Cleaning
The data set was 183 .pkl files broken up by day between 4/1/2018 -9/20/2018. Each file contained 9 columns with many transactions from different customers, the amount, the day and time, and a terminal_id which I assume has something to do with the location of the transaction.  
After combining all the files together here is what we have and the spread in some of the columns.

~~~
# of columns = 9

# of rows = 
# of nulls

Target Balance
# of customer_ids
Spread of amount spent on each day
~~~
## Cleaning, Feature Engineering, Preparing the data for modeling
### Cleaning
-	Rename columns
-	Change datatypes
-	Sorted the data by customer_id and datetime
### Feature Engineering
-	Weekday
```python
  df['Weekday'] = df['Datetime'].dt.day_name()
```
-	Amount of last transaction
```python
 df.loc[df['Customer_id'] == df['Customer_id'].shift(1),'Customer_last_transaction_amount'] = df['Amount'].shift(1)
```
-	Time since last transaction
```python
df.loc[df['Customer_id'] == df['Customer_id'].shift(1),'Time_since_last_cust_transaction'] = round((df['Datetime']-df['Datetime'].shift(1)).dt.total_seconds()/3600,1)
```

### More Cleaning 
Drop columns with:
-		High number of NaN Values
-		High cardinality
Drop columns to prevent leakage and repetition

### Preparing the data for modeling
Because the target variable isn’t balanced, I followed made a mixture of two techniques called random under sampling and random over-sampling.
-	Random under sampling aims to balance class distribution by randomly eliminating majority class examples. This is done until the majority and minority class instances are balanced out. 
-	Over-Sampling increases the number of instances in the minority class by randomly replicating them to present a higher representation of the minority class in the sample.

I randomly eliminated the majority and the minority class until I got a baseline of 66% which was a huge improvement to the __ I had before.
There are some downsides to doing this, but in the end I know there is always a downside with certain techniques and I am willing to accept the consequences and learn from them.

```python
  # randomly pick 10000 yes's
  df_yes = df.loc[df['Fraud'] == 1].sample(10000,replace=True, random_state=35)

  # randomly pick 20000 no's
  df_no = df.loc[df['Fraud'] == 0].sample(20000, replace=False, random_state=35)

  # concat yes and no df together and replace over old df
  df = pd.concat([df_yes, df_no])
```
## After Cleaning
~~~
# of columns = 9

# of rows = 
# of nulls 

Target 
# of customer_ids
Spread of amount spent on each day
~~~
Now we are ready to move on to the modeling
# Models
-
## Logistic
## Decision
## Random Forest

# Tuning Model
## GridSearch
## RandomizedSearchCV

# Importance 
## Logistic
## Decision
## Random Forest
### GridSearch
### RandomizedSearchCV

# Conclusion
