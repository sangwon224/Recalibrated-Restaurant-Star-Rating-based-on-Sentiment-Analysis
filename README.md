# Recalibrated Restaurant Star Rating based on Sentiment Analysis
![image](https://static1.squarespace.com/static/5b1590a93c3a53e49c6d280d/t/5fd058bf5efba8153b21ad7f/1607489731350/restaurant-reviews-16x9.jpg?format=1500w)

## Project Overview
How important is star rating, or online reviews, for a small business, such as restaurant?

According to a study, 88% of consumers read online reviews to decide whether to experience or purchase a business' product or service. And, 94% of consumers will not do a business with a company due to negative online reviews.

Amongst variety of different online review platforms (e.g., Yelp, TripAdvisor, Facebook), Google has the greatest power as it holds ~90% of search traffic.

Considering the aforementioned facts, it is crucial for a business/company to maintain a positive Google rating for its business to thrive.

However, Google's star rating system can be arbitrary as reviewers all hold different criteria for providing certain number of stars. In this regard, this project aims to conduct sentiment analysis to determine whether Google's star rating system is relatively accurate/objective based on the sentiment scores of each review.

## Business Problem
Google Maps Product Manager received many complaints from restaurant owners that Google rating system is quite inconsistent and arbitrary in that many restaurants are either over or under-rated by its customers despite similar quality of food, atmostphere, and service.

For this reason, the PM would like to investigate whether the complaints are valid (or substantiated by data). If the claim is true, the PM is considering updating Google Maps' star rating system by auto-assigning stars based on each customer's review instead of user-assigned star rating system.

To make a data-driven decision, the PM has engaged Sangwon, data scientist, for the analysis.

Note: This business problem is hypothetical; not based on actual complaint

## Data Understanding
### Data Overview
For this project, Google Maps review dataset was pulled from Google Maps API:

- Each API data pull returns ~300 reviews (5 reviews per restaurant) for a given location (e.g., Chelsea)
- Since the Google Maps API provides random ~300 reviews at a time, there may be duplicates in each pull
- Since the API only provides the latest 5 reviews per restaurant, the star ratings composition is imbalanced as user has no control over the review selections
  
Overall, for this project, I have pulled roughly ~3K reviews along with restaurant information to minimize API usage cost and computing time.

### Data Limitations
- Due to the limited computational power, the model will be trained on a relatively small dataset
- The dataset does not have all/full text reviews for a given restaurant (i.e., limited to 5 reviews per restaurant)
- The dataset exclusively examines reviews of New York City restaurants. Different geography may have quite different relationship between text review's sentiment score and star rating

### Data Preparation
Output from the Google Maps API provides quite clean dataset. Hence, minimal data cleansing and pre-processing have been done.

Sentiment analysis model does not require tokenizing, stemming, and lemmatizing as the models rely on pre-built lexicon that contains sentiment scores.

In addition, please note that this notebook was run on Google Colab environment as it was more suitable to run advanced LLM/NLP models from HuggingFace.

## Analysis
For the project, two different sentiment analysis models were used:

- VADER (baseline sentiment analysis model)
- Roberta (sentiment analysis model from HuggingFace, which should be more robust)

Both models provide sentiment score for a given review.

```
VADER Model

sia.polarity_scores(example)

{'neg': 0.0, 'neu': 0.625, 'pos': 0.375, 'compound': 0.9815}
```

```
Roberta Model

encoded_input = tokenizer(example, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dic = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]}
print(scores_dic)

{'roberta_neg': 0.011635479, 'roberta_neu': 0.030706886, 'roberta_pos': 0.95765764}
```

With the sentiment scores for each text review, reviews were clustered based on the sentiment scores to assign recalibrated star rating.

Two clustering methods were used:

- K-Means
- Agglomerative

## Observation
The sentiment analysis models (i.e., VADER and Roberta) did a good job providing general sentiment of a collection of reviews. In fact, it was able to decently distinguish overall sentiment per star rating buckets on an aggregated level. However, for a specific/individual review, the model often provided incorrect sentiment scores, especially in a situation where the text reviews were sarcastic or had nuanced criticisms.

Although the reclassification model often mis-categorized individual reviews in a wrong bucket, on the bright side, we were able to see some occasions where star rating reclassification model correctly reclassified star rating based on text reviews as some users were overly harsh or loose on their star ratings. Hence, we do see an opportunity and benefit of deploying such model.

## Recommendations
From the project, following are my recommendations to the PM:

- Post example reviews per different star rating buckets to create a general criteria/guideline for what each star rating means (to prevent very harsh or loose star ratings)
- Beta-test ML+NLP/LLM-based restaurant star rating system (i.e., based on sentiment analysis model) in addition to the traditional star rating system and get user's feedback
- Beta-test star rating assistant feature: when there appears to be a big mismatch in text review's sentiment score and user's star rating, assistant feature can provide a suggested star rating, which user can override if user would like to

## Future Improvements
- Re-run the analysis and re-train the model with bigger dataset
- Research into more advanced sentiment analysis model that can better pick up nuances (perhaps, utilize ChatGPT 4.0 API)

## For More Information (to be updated)
Please review our full analysis in jupyter notebook ([Recalibrated_Restaurant_Rating_Part 1](https://github.com/sangwon224/Recalibrated-Restaurant-Rating-based-on-Sentiment-Analysis/blob/main/Recalibrated_Restaurant_Rating_Part1.ipynb)) & ([Recalibrated_Restaurant_Rating_Part 2](https://github.com/sangwon224/Recalibrated-Restaurant-Rating-based-on-Sentiment-Analysis/blob/main/Recalibrated_Restaurant_Rating_Part2.ipynb)) \
And also refer to our ([Presentation](--)) 

## Contributors
[Sang-won Shim](https://github.com/sangwon224)

## Repository Structure
```
|— .gitignore                                                <- List of files to ignore
|— README.md                                                 <- Project overview
|— Recalibrated_Restaurant_Rating_Part1.ipynb                <- Jupyter notebook analysis (1/2)
|— Recalibrated_Restaurant_Rating_Part2.ipynb                <- Jupyter notebook analysis (2/2)
|— Data                                                      <- Data folder
    |— NYC_Restaurants_Reviews_(1~11).csv                    <- Raw data
    |— master_df.csv                                         <- csv output of Part 1 analysis
|_ presentation.pdf                                          <- PDF version of project presentation
```
