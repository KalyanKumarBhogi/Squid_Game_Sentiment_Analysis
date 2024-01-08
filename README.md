# Squid_Game_Sentiment_Analysis

The Squid Game is currently one of the most trending shows on Netflix. It is so much trending that people who have never watched any web series before are also watching it. One of the reasons behind this is the reviews and opinions of viewers on social media. So this is how I got interest in doing sentimental analysis on Squid Game.

## Squid Game Sentiment Analysis using Python

### The dataset that I am using for the task of Squid Game sentiment analysis is downloaded from Kaggle, which was initially collected from Twitter while people were actively sharing their opinions about Squid Game. Let’s start the task of Squid Game sentiment analysis by importing the necessary Python libraries and the dataset:

import pandas as pd </p>
import seaborn as sns </p>
import matplotlib.pyplot as plt </p>
from nltk.sentiment.vader import SentimentIntensityAnalyzer </p>
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator </p>

data = pd.read_csv("tweets_v8.csv")
print(data.head())

![image](https://github.com/KalyanKumarBhogi/Squid_Game_Sentiment_Analysis/assets/144279085/4e4e9506-6224-4724-96b7-6ffcf590c1db)
![image](https://github.com/KalyanKumarBhogi/Squid_Game_Sentiment_Analysis/assets/144279085/c5c10f46-bd06-4026-b6c4-f9dd29cfb598)

### In first impressions of this dataset, I noticed null values in the “user_location” column that seem to not affect the sentiment analysis task. So I will drop this column:

data = data.drop(columns="user_location", axis=1) </p>

### Now let’s have a look at whether other columns contain any null values or not:

print(data.isnull().sum()) </p>

![image](https://github.com/KalyanKumarBhogi/Squid_Game_Sentiment_Analysis/assets/144279085/6521f33b-d1eb-44b5-a3e9-8cf0d4efa78f)

### The “text” column in the dataset contains the opinions of the users of Twitter about the squid game, as these are social media opinions, so this column needs to be prepared before any analysis. So let’s prepare this column for the task of sentiment analysis:

import nltk  </p>
import re  </p>
nltk.download('stopwords')  </p>
stemmer = nltk.SnowballStemmer("english")  </p>
from nltk.corpus import stopwords  </p>
import string  </p>
stopword=set(stopwords.words('english')) </p>

def clean(text):  </p>
    text = str(text).lower()  </p>
    text = re.sub('\[.*?\]', '', text)  </p>
    text = re.sub('https?://\S+|www\.\S+', '', text)  </p>
    text = re.sub('<.*?>+', '', text)  </p>
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) </p>
    text = re.sub('\n', '', text)  </p>
    text = re.sub('\w*\d\w*', '', text) </p>
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')] </p>
    text=" ".join(text)  </p>
    return text  </p>
data["text"] = data["text"].apply(clean)  </p>

### Now let’s take a look at the most used words in the Squid Game opinions using a word cloud. A word cloud is a data visualization tool that displays the most used words in a larger size. Here is how you can visualize the word cloud of the text column:

text = " ".join(i for i in data.text)  </p>
stopwords = set(STOPWORDS) </p>
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text) </p>
plt.figure( figsize=(15,10)) </p>
plt.imshow(wordcloud, interpolation='bilinear')  </p>
plt.axis("off") </p>
plt.show() </p>

![image](https://github.com/KalyanKumarBhogi/Squid_Game_Sentiment_Analysis/assets/144279085/f9758240-98fe-4696-a6ac-fed3705058fd)

### Now let’s move to the task of Squid Game sentiment analysis. Here I will add three more columns in this dataset as Positive, Negative, and Neutral by calculating the sentiment scores of the text column:

nltk.download('vader_lexicon') </p>
sentiments = SentimentIntensityAnalyzer() </p>
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["text"]] </p>
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["text"]] </p>
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["text"]] </p>
data = data[["text", "Positive", "Negative", "Neutral"]] </p>
print(data.head()) </p>

![image](https://github.com/KalyanKumarBhogi/Squid_Game_Sentiment_Analysis/assets/144279085/b2be0781-9f0a-401f-a666-96424f7e0146)

### Now let’s calculate how most people think about the Squid Game:

x = sum(data["Positive"]) </p>
y = sum(data["Negative"]) </p>
z = sum(data["Neutral"]) </p>

def sentiment_score(a, b, c): </p>
    if (a>b) and (a>c):  </p>
        print("Positive 😊 ") </p>
    elif (b>a) and (b>c): </p>
        print("Negative 😠 ") </p>
    else: </p>
        print("Neutral 🙂 ") </p>
sentiment_score(x, y, z) </p>

![image](https://github.com/KalyanKumarBhogi/Squid_Game_Sentiment_Analysis/assets/144279085/7a22ea22-4bcd-4f91-8082-814c91b9ba3f)

### So most of the opinions of the users are Neutral, now let’s have a look at the total of each sentiment score before making any conclusion:
print("Positive: ", x) </p>
print("Negative: ", y) </p>
print("Neutral: ", z) </p>

![image](https://github.com/KalyanKumarBhogi/Squid_Game_Sentiment_Analysis/assets/144279085/10c875c8-ba4f-40be-8ff0-bddeae162b7c)

# Summary <p>
The total of negatives is much lower than that of Positive, so we can say that most of the opinions on the Squid Game are positive.
