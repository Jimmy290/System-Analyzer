{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "import numpy as np\n",
    "import matplotlib as plt\n",
    "import pandas as pd\n",
    "from nltk.tokenize import TweetTokenizer\n",
    "from nltk.stem import PorterStemmer\n",
    "from nltk.corpus import stopwords\n",
    "import re\n",
    "import string\n",
    "from nltk.corpus import twitter_samples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "def process_tweets(tweet):\n",
    "    stemmer=PorterStemmer()\n",
    "    stop_words=stopwords.words('english')\n",
    "#     next 2 lines removes noise\n",
    "    tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\\(\\),]|'\\\n",
    "                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', tweet)\n",
    "    tweet = re.sub(\"(@[A-Za-z0-9_]+)\",\"\", tweet)\n",
    "    tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)\n",
    "    tweet_tokens = tokenizer.tokenize(tweet)\n",
    "    clean_tweets=[]\n",
    "    for word in tweet_tokens:\n",
    "        if word not in stop_words and word not in string.punctuation:\n",
    "            stemmed_word = stemmer.stem(word)\n",
    "            clean_tweets.append(stemmed_word)\n",
    "    return clean_tweets\n",
    "            \n",
    "            \n",
    "def build_frequency(tweets, ys):\n",
    "    yslist=np.squeeze(ys).tolist()\n",
    "    frequency = {}\n",
    "    for n2, tweet in zip(yslist, tweets):\n",
    "        for word in process_tweets(tweet):\n",
    "            pair=(word, n2)\n",
    "            if pair in frequency:\n",
    "                frequency[pair] += 1\n",
    "            else:\n",
    "                frequency[pair] =1\n",
    "    return frequency"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "positive_tweets = twitter_samples.strings('positive_tweets.json')\n",
    "negative_tweets = twitter_samples.strings('negative_tweets.json')\n",
    "\n",
    "# split tweets for training and testing.\n",
    "positive_tests=positive_tweets[4000:]\n",
    "positive_train=positive_tweets[:4000]\n",
    "negative_tests=negative_tweets[4000:]\n",
    "negative_train=negative_tweets[:4000]\n",
    "\n",
    "train_n1=positive_train + negative_train\n",
    "test_n1= positive_tests + negative_tests\n",
    "train_n2 = np.append(np.ones((len(positive_train), 1)), np.zeros((len(negative_train), 1))\n",
    ", axis=0)\n",
    "test_n2 = np.append(np.ones((len(positive_tests),1)), np.zeros((len(negative_tests), 1)), axis=0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "12014\n",
      "@Lamb2ja Hey James! How odd :/ Please call our Contact Centre on 02392441234 and we will be able to assist you :) Many thanks!\n",
      "['hey', 'jame', 'how', 'odd', ':/', 'pleas', 'call', 'contact', 'centr', '02392441234', 'abl', 'assist', ':)', 'mani', 'thank']\n"
     ]
    }
   ],
   "source": [
    "frequency = build_frequency(train_n1, train_n2)\n",
    "print(f'{len(frequency.keys())}')\n",
    "\n",
    "print(train_n1[1])\n",
    "print(process_tweets(train_n1[1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sigmoid(n3):\n",
    "    h=1/(1+np.exp(-n3))\n",
    "    return h\n",
    "\n",
    "def gradientDescent(n1, n2, theta, alpha, num_iters):\n",
    "    t = m.shape[0]\n",
    "    for i in range(0, num_iters):\n",
    "        n3 = np.dot(m, theta)\n",
    "        n4= sigmoid(n3)\n",
    "        n5= -1./m*(np.dot(n2.transpose(), np.log(n4)) + np.dot((1-n2).transpose(), np.log(1-n4)))\n",
    "        \n",
    "        theta = theta - (alpha/m)*np.dot(m.transpose(), (n4-n2))\n",
    "        n5= float(n5)\n",
    "        return j, theta\n",
    "    \n",
    "def extract_features(token, frequency):\n",
    "    word_token = process_tweets(token)\n",
    "    m = np.zeros((1,3))\n",
    "    m[0,0]=1\n",
    "\n",
    "    for word in word_token:\n",
    "        m[0,1]+=frequency.get((word, 1.0),0)\n",
    "        m[0,2] +=frequency.get((word, 0.0),0)\n",
    "        assert(m.shape ==(1,3))\n",
    "        return m\n",
    "\n",
    "X = np.zeros((len(train_m),3))\n",
    "for i in range(len(train_m)):\n",
    "    X[i, :]=extract_features(train_m[i], frequency)\n",
    "    \n",
    "Y = train_n2\n",
    "J, theta = gradientDescent(X,Y, np.zeros((3,1)), 1e-9, 1500)\n",
    "    \n",
    "\n",
    "def predict_tweet(tweet, frequency, theta):\n",
    "    m = extract_features(tweet, frequency)\n",
    "    s_pred = sigmoid(np.dot(m, theta))\n",
    "    return s_pred\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.49998551]])"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "my_tweet = 'I am learning :)'\n",
    "predict_tweet(my_tweet, frequency, theta)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Negative\n"
     ]
    }
   ],
   "source": [
    "# custom tweet test\n",
    "\n",
    "custom_tweet = 'best car'\n",
    "tweet_prediction=predict_tweet(custom_tweet, frequency, theta)\n",
    "if tweet_prediction > 0.5:\n",
    "    print('Positive')\n",
    "else:\n",
    "    print('Negative')"
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
