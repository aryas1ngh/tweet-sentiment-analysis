# tweet-sentiment-analysis

This project analyzes tweets posted by random online users. The dataset used was Senitment140, which contains around 1.6 million tweets labelled positive or negative. Other features include date posted and username.

In this project, we perform basic initial visualization on the dataset, followed by cleaning it. This is done ny removing stopwords and punctuation marks. Then, lemmatizing is performed to get root words. For feature extraction, TF-IDF transformer is used instead of CountVectorizer. Finally we show accuracy metrics for the 
model.

The dataset is in `.csv` format and too big to be uploaded directly to GitHub. Here is the Kaggle link to the dataset: [link](https://www.kaggle.com/datasets/kazanova/sentiment140)
