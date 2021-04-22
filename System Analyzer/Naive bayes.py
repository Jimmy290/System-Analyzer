{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],

   "source": [
    "import nltk.classify.util\n",
    "from nltk.classify import NaiveBayesClassifier\n",
    "from nltk.corpus import twitter_samples\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.tag import pos_tag\n",
    "from nltk.stem.wordnet import WordNetLemmatizer\n",
    "import re, string\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.corpus import twitter_samples\n",
    "from nltk import classify\n",
    "from nltk import NaiveBayesClassifier\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "positive_tweets  = twitter_samples.strings('positive_tweets.json')\n",
    "negative_tweets  = twitter_samples.strings('negative_tweets.json')\n",
    "text = twitter_samples.strings('tweets.20150430-223406.json')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# tweet_tokens = twitter_samples.tokenized('positive_tweets.json')\n",
    "# # tweet_tokens[2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### step 1: Normalize the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['@DespiteOfficial', 'we', 'have', 'a', 'listen', 'last', 'night', ':)', 'As', 'You', 'Bleed', 'be', 'an', 'amazing', 'track', '.', 'When', 'be', 'you', 'in', 'Scotland', '?', '!']\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# print(pos_tag(tweet_tokens[1]))\n",
    "\n",
    "def lemmatize_sentence(tokens):\n",
    "    lemmatizer = WordNetLemmatizer()\n",
    "    lemmatized_sentence = []\n",
    "    \n",
    "    for word, tag in pos_tag(tokens):\n",
    "        if tag.startswith('NN'):\n",
    "            pos ='n'\n",
    "        elif tag.startswith('VB'):\n",
    "            pos = 'v'\n",
    "        else:\n",
    "            pos = 'a'\n",
    "        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))\n",
    "    return lemmatized_sentence\n",
    "\n",
    "\n",
    "print(lemmatize_sentence(tweet_tokens[2]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Remove noise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['listen', 'last', 'night', ':)', 'bleed', 'amazing', 'track', 'scotland']\n"
     ]
    }
   ],
   "source": [
    "\n",
    "stop_words=stopwords.words('english')\n",
    "def remove_noise(tweet_tokens, stop_words=()):\n",
    "    cleaned_tweets =[]\n",
    "    for token, tag in pos_tag(tweet_tokens):\n",
    "        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\\(\\),]|'\\\n",
    "                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)\n",
    "        token = re.sub(\"(@[A-Za-z0-9_]+)\",\"\", token)\n",
    "        \n",
    "        if tag.startswith('NN'):\n",
    "            pos='n'\n",
    "        elif tag.startswith('VB'):\n",
    "            pos='v'\n",
    "        else:\n",
    "            pos='a'\n",
    "        lemmatizer=WordNetLemmatizer()\n",
    "        token = lemmatizer.lemmatize(token,pos)\n",
    "        \n",
    "        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:\n",
    "            cleaned_tweets.append(token.lower())\n",
    "    return cleaned_tweets\n",
    "    \n",
    "    \n",
    "print(remove_noise(tweet_tokens[2], stop_words))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "positive_uncleaned_tweets=twitter_samples.tokenized('positive_tweets.json')\n",
    "positive_cleaned_tweets_list=[]\n",
    "negative_uncleaned_tweets=twitter_samples.tokenized('negative_tweets.json')\n",
    "negative_cleaned_tweets_list=[]\n",
    "\n",
    "for words in positive_uncleaned_tweets:\n",
    "    positive_cleaned_tweets_list.append(remove_noise(words, stop_words ))\n",
    "    \n",
    "for words in negative_uncleaned_tweets:\n",
    "    negative_cleaned_tweets_list.append(remove_noise(words, stop_words ))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['@jjulieredburn', 'Perfect', ',', 'so', 'you', 'already', 'know', \"what's\", 'waiting', 'for', 'you', ':)']\n",
      "['perfect', 'already', 'know', \"what's\", 'wait', ':)']\n"
     ]
    }
   ],
   "source": [
    "print(positive_uncleaned_tweets[13])\n",
    "print(positive_cleaned_tweets_list[13])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### step 3: Prepare data model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "#         convert tokens into dictionary for both positiev and negatice tweets\n",
    "\n",
    "def model_for_tweets(cleaned_tokens_list):\n",
    "    for tweets in cleaned_tokens_list:\n",
    "        yield dict([tweet, True] for tweet in tweets )\n",
    "        \n",
    "positive_tweets_model=model_for_tweets(positive_uncleaned_tweets)\n",
    "negative_tweets_model=model_for_tweets(negative_uncleaned_tweets)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "#  split the dataset for training and testing the models\n",
    "\n",
    "import random\n",
    "positive_dataset=[(tweet_dict, 'Positive') for tweet_dict in positive_tweets_model]\n",
    "negative_dataset=[(tweet_dict, 'Negative') for tweet_dict in negative_tweets_model]\n",
    "\n",
    "dataset=positive_dataset+negative_dataset\n",
    "random.shuffle(dataset)\n",
    "\n",
    "train_data=dataset[:7000]\n",
    "test_data=dataset[7000:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### step 4: Building and testing the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy is :  0.9956666666666667\n",
      "Most Informative Features\n",
      "                      :( = True           Negati : Positi =   2065.4 : 1.0\n",
      "                      :) = True           Positi : Negati =    986.0 : 1.0\n",
      "                     sad = True           Negati : Positi =     49.8 : 1.0\n",
      "                   Thank = True           Positi : Negati =     26.3 : 1.0\n",
      "                  THANKS = True           Negati : Positi =     24.4 : 1.0\n",
      "                 welcome = True           Positi : Negati =     22.9 : 1.0\n",
      "                  FOLLOW = True           Negati : Positi =     19.1 : 1.0\n",
      "                    THAT = True           Negati : Positi =     19.1 : 1.0\n",
      "                    miss = True           Negati : Positi =     18.5 : 1.0\n",
      "                    MUCH = True           Negati : Positi =     18.4 : 1.0\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "from nltk import classify\n",
    "from nltk import NaiveBayesClassifier\n",
    "classifier = NaiveBayesClassifier.train(train_data)\n",
    "\n",
    "print(\"Accuracy is : \", classify.accuracy(classifier, test_data))\n",
    "print(classifier.show_most_informative_features(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
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
    "custom_tweet=\"\"\"\n",
    "During opening statements in Derek Chauvin's trial in the death of George Floyd, prosecutors showed jurors a bystander video of the former officer kneeling on Floyd's neck.\n",
    "\"\"\"\n",
    "custom_tokens = remove_noise(nltk.word_tokenize(custom_tweet))\n",
    "print(classifier.classify(dict([token, True] for token in custom_tokens)))"
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
