# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: RITZ LONGJAM

*INTERN ID*: 

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

#TASK DESCRIPTION: Sentiment Analysis of Customer Reviews Using TF-IDF and Logistic Regression

In this task, we performed sentiment analysis on a large dataset of customer movie reviews to classify them as either positive or negative. Sentiment analysis is a core Natural Language Processing (NLP) technique that is widely used in industries for understanding customer opinions, detecting emotions in text, and automating content moderation. This task demonstrates how machine learning can be used to derive insights from textual data using Python programming in Google Colab.

We began the task by choosing the IMDb Movie Reviews Dataset available on Kaggle, which contains 50,000 labeled reviews categorized into positive and negative sentiments. The dataset was downloaded directly into our environment using the Kaggle API. Google Colab served as the development platform for this project due to its ease of use, powerful computing resources, and integration with cloud services such as Kaggle.

To work with the dataset, we used essential Python libraries such as pandas, numpy, scikit-learn, matplotlib, and seaborn. Pandas was used for data manipulation, NumPy for numerical operations, and scikit-learn for the entire machine learning workflow—from preprocessing to evaluation. Seaborn and Matplotlib helped in creating visualizations like class distribution and confusion matrix.

Before building the model, we performed text preprocessing to clean the reviews and make them suitable for analysis. 

This included:

•	Converting text to lowercase to ensure uniformity 

•	Removing HTML tags using regular expressions, as they don’t contribute to the sentiment.

•	Eliminating punctuation and numbers, which usually add noise and don't affect sentiment.

•	Stripping extra whitespace from the beginning and end of each review to clean formatting.

After cleaning the text data, we converted the text into a numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. TF-IDF is a feature extraction method that reflects how important a word is in a document relative to the entire corpus, and it is commonly used in NLP tasks. We limited the features to 5000 terms to maintain performance and reduce overfitting.

Next, we split the data into training and testing sets (80/20 ratio) and trained a Logistic Regression model, which is a popular baseline algorithm for binary classification problems. After training, we used the model to predict sentiments on the test set.

To evaluate the model’s performance, we measured accuracy, printed a classification report (which includes precision, recall, and F1-score), and visualized the confusion matrix. The model demonstrated reliable performance in classifying sentiments, proving that even a simple linear model like logistic regression can be effective when combined with proper feature engineering like TF-IDF.

Real world applications:

•	E-commerce platforms can analyze customer reviews to assess product satisfaction.

•	Movie streaming services can use sentiment analysis to understand viewer opinions.

•	Social media monitoring tools can classify public reactions to news or brand promotions.

•	Customer support systems can prioritize negative reviews for escalation.

In summary, this task highlighted the entire workflow of a text classification problem—from loading data, preprocessing, and feature extraction, to training and evaluating a machine learning model. Using Python in Google Colab, along with scikit-learn and TF-IDF, we demonstrated how sentiment analysis can be performed efficiently and effectively on real-world textual data.

*OUTPUT*

<img width="412" alt="Image" src="https://github.com/user-attachments/assets/98d6a8a3-5e49-44b5-ae6b-a6aabcc3930d" />
