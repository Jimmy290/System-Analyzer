{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "import time\n",
    "from sklearn import svm\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "train_data_csvsrc=\"https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv\"\n",
    "test_data_csvsrc =  \"https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/test.csv\"\n",
    "\n",
    "train_data = pd.read_csv(train_data_csvsrc)\n",
    "test_data = pd.read_csv(test_data_csvsrc)\n",
    "\n",
    "vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)\n",
    "\n",
    "train_vectors = vectorizer.fit_transform(train_data['Content'])\n",
    "test_vectors = vectorizer.transform(test_data['Content'])\n",
    "\n",
    "classifier_linear = svm.SVC(kernel='linear')\n",
    "t0 = time.time()\n",
    "classifier_linear.fit(train_vectors, train_data['Label'])\n",
    "t1 = time.time()\n",
    "prediction_linear = classifier_linear.predict(test_vectors)\n",
    "t2 = time.time()\n",
    "time_linear_train = t1-t0\n",
    "time_linear_predict = t2-t1\n",
    "\n",
    "# print(f'training time: {time_linear_train}, prediction time: {time_linear_predict} ')\n",
    "report  = classification_report(test_data['Label'], prediction_linear, output_dict=True)\n",
    "print(report['pos'])\n",
    "print(report['neg'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Testing with custom tweet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['pos']\n"
     ]
    }
   ],
   "source": [
    "#  add custom tweet here \n",
    "custom_tweet=\"\"\"\n",
    "    #add custom tweet here\n",
    "\n",
    "\"\"\"\n",
    "custom_vector = vectorizer.transform([custom_text])\n",
    "print(classifier_linear.predict(custom_vector))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
