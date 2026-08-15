# IMDB Movie Review Sentiment Analysis

## Why I built this

I wanted to dig into NLP properly — not just run a pre-built sentiment classifier, but actually compare how traditional machine learning approaches stack up against a modern transformer model on the same task, and see if the extra complexity of something like RoBERTa is actually worth it.

## The data

I used the IMDB Dataset of 50K Movie Reviews — a well-known NLP benchmark dataset with 50,000 reviews, split evenly between positive and negative sentiment (25,000 each), so no class imbalance issues here, which was a nice change from some of my other projects.

## The business angle

This isn't just a classification exercise — automatically sorting reviews into positive/negative has real uses: recommending similar movies based on audience reaction, helping studios spot what's actually driving negative reviews, and targeting marketing toward films that are already resonating with audiences.

## What I did

**Cleaned and preprocessed the text.** Stripped out HTML tags, URLs, and non-ASCII characters, lowercased everything, then tokenized and lemmatized the reviews (reducing words to their base form) and removed stopwords using NLTK. Text data needs a lot more cleanup than tabular data — this step alone took real thought.

**Explored the data first.** Checked class distribution (perfectly balanced, 25K/25K) and built word clouds for positive vs. negative reviews separately, which was actually a nice way to sanity check that the sentiment labels made sense before modeling anything.

**Converted text to numbers two ways** — Bag-of-Words and TF-IDF, both capped at the 5,000 most frequent words to keep things manageable.

**Started with traditional ML models:**

| Model | CV Accuracy | Test Accuracy | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 84.8% | 85.1% | 0.927 |
| Naive Bayes | 84.3% | 84.2% | 0.921 |

Both did solidly well right out of the gate, honestly better than I expected from TF-IDF + a linear model.

**Then tried a transformer model** — fine-tuned RoBERTa (via simpletransformers) on the same data, to see if the extra complexity was actually worth it.

| Model | Accuracy |
|---|---|
| Logistic Regression | 85.1% |
| Naive Bayes | 84.2% |
| RoBERTa (Transformer) | **87.6%** |

RoBERTa won, which makes sense — it can pick up on context and word order in a way TF-IDF just can't. But it's worth saying: the gap between the transformer and the simple Logistic Regression model isn't huge (about 2.5 points), while the transformer takes dramatically more compute and training time. That tradeoff is worth thinking about depending on what you're actually building — a lightweight app that needs to run fast and cheap might be totally fine with Logistic Regression, while something where accuracy really matters might justify the extra cost.

## What I learned

The biggest thing this project taught me: bigger/fancier models aren't automatically the right call. Logistic Regression on TF-IDF features is fast, cheap, easy to explain, and got within a couple points of a transformer that took way more resources to train. Knowing when the extra complexity is actually worth it feels like a more useful skill than just knowing how to fine-tune a transformer.

## What I'd do differently next time

- Try the full dataset instead of a 10% sample for the transformer training — I sampled down mainly for training time, but the full dataset would likely close the gap even further or confirm the transformer's edge is real
- Test other transformer architectures (BERT, DistilBERT) to see how they compare on both accuracy and training time
- Look at where each model actually gets things wrong — the reviews that fool Logistic Regression but not RoBERTa (and vice versa) would tell a more interesting story than just the accuracy numbers

## Tools
Python, pandas, NLTK, scikit-learn, TF-IDF/Bag-of-Words, simpletransformers (RoBERTa), matplotlib, seaborn, WordCloud
