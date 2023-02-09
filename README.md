In this dataset, we will use the content of “body" for training and "subreddit" as the labels for classification machine learning. To make it easier to perform statistics on the dataset, I first used the json_normalize function to transform the json data structure into a dataframe structure from the pandas library. By calling the value_counts function in the dataframe, we will get the counts for the various labels. There are nine different labels in this dataset which includes PS4, pcgaming, NintendoSwitch, antiMLM, HydroHomies, xbox, Coffee, tea, Soda.

Below table will show the counts of the various labels in the training set/validation set/test set.



Table 1: The Labels Counts in Train Set/Validation Set/Test Set

In order to more intuitively understand the distribution  of different labels, I have drawn their distribution histograms separately.



 

**Q1b:**

The model trained by the Dummy classifier can be used as a baseline for other models. In this question we implement two different Dummy classifiers, most_frequent and stratified. most_frequent always returns the most frequent class labels in the training set, and stratified will randomly return class labels according to the prior probabilities. Since we concluded in Q1 that there is a gap between the label distributions in the training and test sets, we chose to use the stratified model of the Dummy classifier as our baseline. 

![img](file:////Users/dongzhanxie/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image008.jpg)

Table2: Performance of five different models

Table 2 points out that the three machine learning models we used, including LogisticRegression with One-hot vectorization, LogisticRegression with TF-IDF vectorization, and SVC Classifier with One-hot vectorization, all achieved significant performance improvements compared to baseline. This demonstrates that all three machine learning models have a better fit to both the training and test sets.

In practice, however, there is still some space for improvement in these three models due to the limitations of the dataset. There are two main problems concerning the dataset. The first problem, mentioned in Q1a, is that the distribution of labels in the training and test sets is quite different. One solution was to reset the training and test sets in the same proportion. In addition, the other major problem was that the dataset was too small, with only 2,000 posts in total, which prevented the model from being adequately fitted. The solution was to switch to a larger dataset.

By observing the values of precision, recall, etc. for each model, we can find that among the three machine learning models, LogisticRegression with TF-IDF vectorization has the highest values for various evaluation values, i.e. it is the best performing model. The bar chart below shows the F1 scores for each label in LogisticRegression with TF-IDF vectorization.



In fact, LogisticRegression with One-hot vectorization and SVC Classifier with One-hot vectorization, both of which use the One-hot Vectorizer, perform similarly to each other. Therefore I concluded that between the three models, the different Vectorizer choices most affected the final performance of the model, rather than the choice of Classifier. However, since the classifiers I used in Q1b were in their default configuration. Perhaps, after parameter tuning, different classifiers would reflect a wide gap in performance. The reason for the good performance of the TF-IDF Vectorizer is that compared to the binary representation of the One-hot Vectorizer, the TF-IDF carries more feature information, which brings a big boost for training machine learning models.

**Q1c:**

I have chosen to use K-Nearest Neighbors as our classifier, which is a classical classification algorithm that selects a different number of nearest neighbours by choosing different values of K as our training method. We implement this algorithm by means of calls to the sklearn library. Through some experimentation in my problem, I eventually chose to use a K value of 10 as our parameter choice.

The Vectorizer TF-IDF, which performed better in Q1b, was chosen as our Vectorizer. Term frequency means the number of times the word appears in the document, IDF means the logarithmically scaled inverse fraction of the number of documents that contain the word. Using TF-IDF performs better than simply using One-Hot. In this problem, I called the sklearn library to implement the vectorize for TF-IDF.

**Q2a:**

Parameter tuning is a simple and effective way to improve the accuracy of a machine learning model. Both the Classifier and Vectorizer in the Sklearn library provide many kinds of parameters that can be adjusted to suit different machine learning scenarios. Experiment with various combinations of parameters and evaluate their performance to obtain a better machine learning model. As compared to manually trying different parameters and evaluating them, the sklearn library provides a more efficient way of parameter tuning, i. e. using a combination of Pipline and GridSearchCV. GridSearchCV will require a dictionary containing the parameters and the values to be selected for each parameter. GridSearchCV will automatically arrange the parameters and evaluate their results, returning the best performing parameter among them. However, if GridSearchCV has too many parameters to be selected, the number of evaluations required will be much higher and will take a very long time. Therefore we need to filter the parameters before using GridSearchCV.

The parameters I chose to tune included the value of C and their solver for the logistic regression model, and sublinear_tf, max_features for the tfidf Vectorizer. 

**sublinear_tf** is a binary parameter which indicates whether to apply sublinear tf scaling or not. As this parameter is binary, we do not need to filter the parameters in advance.

**max_features** is a parameter that limits the maximum vocabulary size. While not limiting the size of the vocabulary will provide more information to the model, too large a vocabulary size may cause over-fitting. We need to filter the values to be selected for the max_features parameter in advance. We provide a figure of F1 Score with different max_features. The figure shows that after max_features reaches 2000 there is no significant increase in F1 scores so for max_features, I will choose four parameters [1900, 2000, 2100, None].

![形状  中度可信度描述已自动生成](file:////Users/dongzhanxie/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image012.jpg)

Figure 3 F1 Score with different max_features

**solver** is a parameter choose which algorithm will use in the LogisticRegression model. I will choose four parameters ['lbfgs', 'liblinear', 'sag', 'saga'].

**C** is a parameter which means the inverse of regularization strength. The figure shows that after C reaches 0.6 there is no significant increase in F1 scores, so for C I will choose three parameters [0.4, 0.5, 0.6].



The parameters we obtained by using GridSearchCV eventually is **C = 0.6; solver = saga; max_features = 2100; sublinear_tf = True.** 

|                             | LogisticRegression with TF-IDF without tunning | LogisticRegression with TF-IDF with tuning |
| --------------------------- | ---------------------------------------------- | ------------------------------------------ |
| Macro-averaged precision    | 0.766                                          | 0.777                                      |
| Weighted-averaged precision | 0.762                                          | 0.772                                      |
| Macro-averaged recall       | 0.760                                          | 0.762                                      |
| Weighted-averaged recall    | 0.755                                          | 0.760                                      |
| Macro-averaged F1           | 0.760                                          | 0.766                                      |
| Weighted-averaged F1        | 0.756                                          | 0.763                                      |

Table3: Performance of LogisticRegression with TF-IDF with and without tuning.

The table indicates that there is some improvement in the performance of the model after tuning the parameters. Restricting the maximum vocabulary size improved the performance of the model, suggesting that too many terms caused overfitting. The overfitting was also reduced by regularising the model with C-values. Besides, saga algorithm performed better for multi-category classification. Sublinear tf scaling of the features also improved the performance of the model.

**Q2b:**

The model still shows some errors after adjusting the parameters, and the error analysis will be carried out from two different aspects: Data set and the Model.

**Data Set:**

\1.   Differences in the distribution of the test and training sets. This causes a Dataset Shift, which leads to a reduction in the robustness of the model on the test set.

\2.   The data set was too small, which resulted in not having enough data for the model to be successfully fitted.

\3.   I found NintendoSwitch, PS4, pcgaming, and xbox to be relatively inaccurate compared to the other labels. One possible reason for this is that many of the same terms may appear in these four labels. For example, a game may be released on all four platforms at the same time. Therefore, based on this game as a term, we can not determine effectively which labels it belongs to.

**Machine Learning Model:**

\1.   Our model appears to be overfitting. When our model is applied to the training set, our model is exceptionally accurate, but when applied to the test set, the accuracy drops significantly. This means that our model was overfitted to the test set.

\2.   When using TF- IDF for Vectorizer, the input do not eliminate Stopwords, which are not helpful for classification. Instead, their presence affects the weight of the logistic regression model.



 

**Q3a:**

I will be using a logistic regression model as the classifier, using TF-IDF as the vectorizer, and adding the following two features to this strategy.

\1.   Feature 1: Add other properties of the posts

I found that our post was only using the body property but never using title and author, which would have included a lot of useful terms. This is certainly a waste of data. Therefore, I merged title, body and author before using TF-IDF Vectorizer, thus making use of all three attributes at the same time.

\2.   Feature 2: Improvement on Vectorizer

I used spicy for natural language processing and added stopwords from the nltk library. Terms were made more efficient by removing stopwords from the text.

**Q3b:**

The table below shows the evaluation results for LR with TF-IDF with tuning, LR with TF-IDF with feature1 added and LR with TF-IDF with feature2 added.

|                             | LogisticRegression with TF-IDF with tuning | LogisticRegression with TF-IDF vectorization with  feature 1 | LogisticRegression with TF-IDF vectorization with  feature 2 |
| --------------------------- | ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Macro-averaged precision    | 0.777                                      | 0.837                                                        | 0.858                                                        |
| Weighted-averaged precision | 0.772                                      | 0.833                                                        | 0.857                                                        |
| Macro-averaged recall       | 0.762                                      | 0.827                                                        | 0.858                                                        |
| Weighted-averaged recall    | 0.760                                      | 0.825                                                        | 0.853                                                        |
| Macro-averaged F1           | 0.766                                      | 0.830                                                        | 0.856                                                        |
| Weighted-averaged F1        | 0.763                                      | 0.827                                                        | 0.853                                                        |

Table4: Performance of LogisticRegression tuning, added feature1, added feature2

In the table we can see that the accuracy of the model improves substantially by 6% after adding title and author. The model also improved after the addition of stopword exclusion.

**Q3c:**

The HotMap below shows the predictions for each label under different strategies. We found that the prediction accuracy of labels such as PS4, pcgame, etc., which had previously performed poorly, was greatly improved after adding my own features.

**![图片包含 文本  描述已自动生成](file:////Users/dongzhanxie/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image016.jpg)![图片包含 文本  描述已自动生成](file:////Users/dongzhanxie/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image018.jpg)![图片包含 文本  描述已自动生成](file:////Users/dongzhanxie/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image020.jpg)**

Figure 5 HotMap in different features

We can conclude from Table 4 and Figure 5 that both of the features we have added significantly improve the performance of the model. 

Performance improved significantly with the use of the post's title and author attributes. The main reasons for this situation include that most titles summarize the article. Therefore, the title can be regarded as a manual feature extraction of the important content of the post. Terms provided by titles will provide a significant influence on the weight of the logistic regression model. Besides, it is quite possible for the same author to write multiple posts under the same subreddit. The inclusion of the author attribute will also help to improve the performance of the model.

Removing stopwords after nlp processing of text using the spacy library also improves the accuracy of our LR models. The stop word will help humans to understand the structure of the text but is confusing for machine learning models. An excessive number of stopwords can affect the model's weight. By removing these stopwords, the model will focus more on the more important information.
