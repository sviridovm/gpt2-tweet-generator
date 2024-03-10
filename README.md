# GPT2-TWEET-GENERATOR

Simple Library for fine tuning a gpt2 model on text data, particularly twitter tweets.

### Create a datset like this

```python
    tweet_dataset = Dataset('tweet_data.csv')
    tweet_dataset.clean_data(Links = True, hashtags = False, mentions = False)

    tweets = tweet_dataset.get_data()
```

### Train your GPT2 Model like this

```python
    model = model('model1')
    model.init_data_loader(tweets)
    model.init_optimizer()
    model.train(epochs=3)


    print(model.generate_text('right or left twix'))
```
