
# Spotify Sequential Skip Prediction Challenge: a smart and light approach
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/music-light-headphone.jpg)

**The right data can be more cost effective than big data.**

**The model trained by the data of 1 day before can best predict the skipping behavior on Spotify.**

## 0. Intro
Spotify is one of the most popular online music streaming services, with hundreds of million users and more than 40 million music tracks. Recommending the right music track to the right user at the right time is their key to success.

In 2018, Spotify released [an open dataset of sequential listening data](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge). According to them, “_a central challenge for Spotify is to recommend the right music to each user. While there is a large related body of work on recommender systems, there is very little work, or data, describing how users sequentially interact with the streamed content they are presented with. In particular within music, the question of if, and when, a user skips a track is an important implicit feedback signal._“

## 1. Dataset
This is the description from their [website](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge): “_The public part of the dataset consists of roughly 130 million listening sessions with associated user interactions on the Spotify service. In addition to the public part of the dataset, approximately 30 million listening sessions are used for the challenge leaderboard. For these leaderboard sessions the participant is provided all the user interaction features for the first half of the session, but only the track id’s for the second half. In total, users interacted with almost 4 million tracks during these sessions, and the dataset includes acoustic features and metadata for all of these tracks._”

“_The task is to predict whether individual tracks encountered in a listening session will be skipped by a particular user. In order to do this, complete information about the first half of a user’s listening session is provided, while the prediction is to be carried out on the second half. Participants have access to metadata, as well as acoustic descriptors, for all the tracks encountered in listening sessions._“

“_The output of a prediction is a binary variable for each track in the second half of the session indicating if it was skipped or not, with a 1 indicating that the track skipped, and a 0 indicating that the track was not skipped. For this challenge we use the **skip_2** field of the session logs as our ground truth._“

See the description of the [dataset schema](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/references/Dataset%20Description.pdf) and the details of the [audio features](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/references/acousticness.pdf).

## 2. Goal: a light & smart predictor
I plan to follow the official challenge as much as I can. Although the competition was over so I have no access to their official validation dataset, I made the first 10,000 sessions of the file `log_0_20180918_000000000000.csv`, one of the file of the final date of the entire dataset (from July 15 to Sep 18, 2018), as my validation data for this machine learning model. In the validation dataset, while the listening log, including skipping behavior (‘skip_2’), is fully available for the **first half** of the listening sessions, the goal is to **predict the skipping behavior on each track in the second half** of each session.

Another goal I set for myself is to **build a powerful predictor while using as little data as possible**. It has two benefits: First, this model can be much lighter and easier to train, to maintain, and to deploy. Second, this model can be useful to predict rare or new emerging patterns. These benefits are particularly relevant for this case, as hundreds of new music tracks are published everyday, and people’s music preference can change from day to day.

Rather than feeding data up to hundreds of GB into a bulky deep learning model running on a high performance computer, to achieve the goal of building a light machine learning model, I need to do it in a smart way. The features will be carefully engineered and augmented based on the domain knowledge, and the model will be carefully curated.

## 3. Approaches & data engineering
I plan to train a machine learning model based on 3 types of data: **Summarized data** of the 1st half tracks per session, **Recommender data** based on all listening behavior, and the **Track data** in the 2nd half of each session. Each of them contains different aspects.
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/Spotify_data_processing%20-%20Data%20preprocessing%402x.png)
- **Summarized data**: The data in the 1st half of each session will be summarized as mean, standard deviation (std) and median among the skipped or non-skipped tracks in a session. For instance, the loudness variable will be augmented into 6 variables: “the mean/std/median loudness among the skipped/non-skipped tracks”, and each augmented variable will be the same information for all tracks within the same session.
- **Recommender data**: Collaborative filtering will be estimated based on the skipping behaviors of many session on many tracks. It will be used to generate unique recommendation value per track per session.
- **Track data:** The model will predict each track will be skipped or not. It contains same track features at different position in different sessions.
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/Spotify_data_processing%20-%20data%20features%402x.png)

## 4. Exploratory data analyses (EDA)
[Jupyter notebook of EDA](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/0_exploratory_data_analysis.ipynb)

In general, the skipping rate was around 50%. I further plotted the skipping rate by session position (among the 2nd half of each session). Although the general skipping rates were between 46% and 54%, still pretty close to 50%, there was a trend that the skipping rate drops from position 6 to 10 and then increases monotonically. This trend might somehow be an artifact of the fact that, in this dataset, almost half of the sessions had 20 tracks and the remaining sessions had between 10 and 19 tracks. Despite the reason of this pattern was unclear, as I don’t know how Spotify curated the data, it suggested that (1) session length and session position can be good predictors, and (2) the skipping behavior can be considered as a balanced dataset for machine learning.
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/skip_position.png)

I used paired t-tests to quickly examine what variables were different between skipped and non-skipped tracks.
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/tvalues_skipped_or_not.png)

It appeared that the **listening behaviors (reasons to start and end a track) are closely associated with the skipping behavior**. Would it be a circular reasoning that a track was skipped because the user had a reason to skip it? Possible! Would it be a data leakage? No, because the listening behaviors will only be included in the summarized data but not the data to be predicted!

Another interesting observation is that the **track features had relatively low associations with the skipping behavior**. It makes sense as skip_2 means that a user skips a track after only listening for a brief period. However, the track features were obtained by analyzing the entire track. Therefore, the track features have not been perceived by the users at the moment they skip it, so it could have low association with skipping or not.

## 5. Recommender system
[Recommender system](https://en.wikipedia.org/wiki/Recommender_system) is a popular approach to recommend specific items to specific users based the past user-item interaction, in our case, whether a track was skipped or not in a listening session.

There are three common approaches to build a recommender system:

- **Content-based filtering**: If a user skips Track A, he/she is likely skip other tracks which share the similar features as Track A.
- **Collaborative filtering**: Given that User X skips Track A, B and C, and User Y skips Track A and B, considering the similar skipping behavior between these two users, User Y is likely to skip track C too as User X did. See [Google’s webpage](https://developers.google.com/machine-learning/recommendation/collaborative/basics) for more in-depth explanation.
- **Hybrid**: According to Wikipedia, “there is no reason why several different techniques of the same type could not be hybridized.”
I used the hybrid approach with the following steps:

### 5-1. Use k-means clustering to group the track dataset
[Jupyter notebook of clustering](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/1_clustering_tracks.ipynb)

It can be considered as a content-based filtering as the clustering was performed based on the track features. It benefits the following common issues of collaborative filtering:

- **Cold start**: As there isn’t enough data from the new sessions and tracks to do collaborative filtering, grouping them with the other existing tracks with similar features can make them “not new” and use the existing data.
- **Sparsity**: Given that there was max 20 tracks per session in the current dataset, but there are 3,706,388 tracks in the data. Therefore, the overlap between tracks and sessions will be very sparse if not using clustering.
- **Scalability**: The dataset is way too big (> 350 GB) to be computed on any single local machine. Grouping the tracks into clusters can largely reduce the data dimension and speed up the computation.
However, it comes with a price that all the tracks within the same cluster will be considered the same for the recommender system.

Mini-batch k-means clustering uses small random batch of data to do k-means clustering. To determine the parameter k, the inertia by k was plotted here:

![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/kmean_inertiaPlot.png)

It appears that the elbow location was around 200. But to preserve the fine difference among the tracks as much as possible, k=300 was used.

![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/KMean300_distribution.png)

The number of tracks within each cluster ranged from ~1,000 to 25,000. It nicely reflects a fact that some music styles/genres are more popular than others.

### 5-2. Data reduction using singular value decomposition (SVD)

[Jupyter notebook of SVD](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/2_collaborative_filtering_svd_allTracks.ipynb)

The data for doing collaborative filtering should be a matrix of session * tracks. The k-means clustering has reduced the number of tracks from 3 millions to 3 hundreds, but there are still hundreds of thousands of sessions within each data file. To further reduce the data size to make it computationally feasible, I used [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) (M = UΣV*) to reduce the matrix (M) from >1,000,000x300 to 300x300 (V*).

SVD is computationally very intensive to implement. So, I used the dask module for parallel computing.

### 5-3. Calculating similarity or distance using multiple metrics
[Jupyter notebook for calculating similarity](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/3_collaborative_filtering_similarities_for20180918_mutipleRanges.ipynb)

Cosine similarity is a popular approach to calculate the similarity between items (or the track clusters in this case). But why not calculate as many similarity/distance metrics as possible? Therefore, I ended up calculating 9 metrics using scipy, including cosine similarity, Pearson correlation, Spearman correlation, Kendall correlation, Manhattan distance, Canberra distance, Hamming distance, Chebyshev distance, and Bray–Curtis dissimilarity. (I also calculated Euclidean and squared Euclidean distances before I realized that all the paired distance will be exactly the same as the matrix V* is a unitary matrix.)

## 6. Preprocessed data
After all the data preprocessing, there are 308 features * more than 30 millions rows of data per day for training.

## 7. Training and tuning LightGBM
[Jupyter notebook of Bayesian parameter tuning](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/4_LightGBM_BayesianOpt_for20180918.ipynb)

[Jupyter notebook of fitting LightGBM](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/5_LightGBM_fit_multiDays_for20180918.ipynb)

LightGBM is a popular gradient boosting algorithm well-suited for modeling tableau data. It was chosen over XGBoost and CatBoost because of its fast speed while remaining high accuracy ([Bentéjac et al., 2020](https://doi.org/10.1007/s10462-020-09896-5)).

I used [**`BayesianOptimization`**](https://github.com/fmfn/BayesianOptimization) to tune the parameters of LightGBM because of its fast speed. This iterative algorithm based on bayesian inference and gaussian process makes it much efficient than the traditional grid-search or random-search approaches. [Here](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) and [here](https://docs.aws.amazon.com/sagemaker/latest/dg/lightgbm-tuning.html) are some general recommendations on the ranges of each parameters.
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/bayesian_optimization.gif)
Source: https://github.com/fmfn/BayesianOptimization

Note that as Spotify announced [a customized evaluation metrics AP](https://github.com/crowdAI/skip-prediction-challenge-starter-kit) (or AA, as they used both terms interchangeably), which is conceptually a conditional accuracy of predictions in a sequence.
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/AP_formula.png)

n is the length of the sequence, yi is the boolean outcome of the prediction being correct or wrong (1 for correct, 0 for wrong).

For instance, if the predictions of a sequence of 5 tracks are [1, 1, 1, 0, 1] , then the AP equals to (1/1 + 2/2 + 3/3 + 3/4 + 4/5)/5 = 0.910. For another instance: [0, 1, 1, 1, 1], the AP would be (0/1 + 1/2 + 2/3 + 3/4 + 4/5) = 0.543. Despite these two sequences have the same mean accuracy 0.8, their **AP values are quite different, as AP puts more weights on the earlier tracks than the later tracks**. Therefore, this factor needs to be considered while training the machine learning model.

While it is possible to use this customized metrics to train the model, this iterative calculation is extremely slow. To speed up, I used this script to empirically estimate the weights of the tracks in a sequence as an approximation. These estimated weights can simply be incorporated into any machine learning models.

## 8. Prediction performance
[Jupyter notebook of exploring data sizes](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/6_LightGBM_predict_multipleDays_for20180918.ipynb)

### 8-1. The data of how many days should be included?
As stated earlier, one goal of this project is to explore how little training data should be used for training this model. Therefore, I trained the model with different numbers of previous days’ data, from 1 day to 1 week.
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/ap_by_days.png)

Although all the model performances were well above the baselines, it was clear that the **prediction performance was the best when only including the training data from 1 day before**. The performance dropped significantly when the training data spanned to 2 or 3 days before. The model performance increased again when adding more older data, the but performance did not outperform the 1-day model.

A possible explanation is that the information from the days further away are more different to the validation dataset. Therefore, adding 2 or 3 days’ of data before the validation dataset into the model was like adding noise. This situation could be mitigated when adding sufficient amount of older data, as some subtle patterns might emerge with sufficient amount of “noise”. However, the benefit of adding more older data can grow very slowly, and the performance of 7-day data not yet reached the level of 1-day data.

Note that there are more than 30 million rows of data each day, and there could be some qualitative difference from day to day due to various real-world events, such the announcement of a popular song, or people were suddenly interested in an old song appeared in a commercial ad on TV. **A better machine learning model should allow data scientists to manually factor in any potentially important real-world events.**

### 8-2. Model evaluation
For the model with the best performance, the **AP was 0.571**, and the **mean accuracy of the 1st track was 0.744**.

Also, compared to the [leaderboard](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/leaderboard.ipynb) of this open challenge, the current best model (1-day) ranked within the top 1/3 of all the submissions.

![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/leaderboard.png)

This model was not overfitted. The mean accuracy in the cross-validation was 0.721, and the mean accuracy of the validation dataset was 0.677, **the magnitude of overfitting was merely 4.4%**.

## 9. What features contribute to the prediction?
[Jupyter notebook for visualization](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/notebooks/7_LightGBM_SHAP_visualize_for20180918.ipynb)

[SHAP](https://shap.readthedocs.io/en/latest/) (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. Conceptually, it explores how much the model prediction will change after adding each variable.
![](https://github.com/curlsloth/Capstone_Spotify-Sequential-Skip-Prediction/blob/main/reports/figures/SHAPsummary1_boost_alltracks_incrementalTrain_10_multidays_jan14.png)

This plot summarized the top 40 most influential features of the current model.

An interesting observation is that **34 out of 40 top features were the augmented summarized data of each session’s history**. This is consistent with the impression from the EDA. Take the `session_position_sk2True_mean` for instance, if the later tracks in the 1st half of a session were skipped more often, the following tracks in the 2nd half of the same session were likely being skipped too (in other words, users were skipping multiple tracks in a row).

There were 3 out of 9 recommender features in the top-40 list, which were the recommender based on Kendall correlation, Hamming distance, and Spearman correlatin. For instance, the `KendCorr300` feature in the plot means the observations with lower values of `KendCorr300` (more blue) are more likely to be predict it as `skip2==True` (higher SHAP value). It makes sense as lower `KendCorr300` means the track in the an observation is less similar with the preferred tracks in the 1st half of the same session. It suggested that the individual recommender data is very helpful, and including multiple similarity/distance metrics can be helpful.

Last, only one track feature, the `us_popularity_estimate` was in the top-40 list. It suggests two things: (1) Different users prefer different music, and there isn’t a feature that will make everyone dislike (otherwise it would not be published). (2) Popularity is essentially a metrics of all users’ preference, which is conceptually similar to collaborative filtering. It explains why this was the only important track feature in this model.

## 10. Conclusion
This project demonstrated that **the right data is more useful than big data**. I built a light and fast machine learning model with even very limited amount of data, even just a day before. Considering the costs of curating and computing big data, **proper data engineering on the right data guided by domain knowledge can be more cost effective**.

So, what can the stakeholders do to make the Spotify users to skip less? Despite many factors cannot be controlled by Spotify (such as the summarized data of each session’s history), (1) **building a more precise recommender system based on multiple similarity/distance metrics** and (2) recommend the tracks which have **higher general popularity** will lower the skipping rate.

