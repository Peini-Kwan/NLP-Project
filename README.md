# Sentiment Analysis and Topic Modeling of The North Face Store on Amazon E-commerce
## Project Overview
This project aims to perform sentiment analysis and topic modeling on customer reviews of The North Face store on Amazon. By analyzing customers' sentiment and identifying key topics, the intention of this project is to gain valuable insights into the opinions and preferences of customers regarding The North Face products available on the e-commerce platform.
## Data Collection
A dataset of customer reviews is collected from The North Face store, under Men's Section on Amazon using web scraping techniques. The dataset consists of textual reviews along with corresponding ratings and other metadata. Before analysis, data preprocessing steps such as text cleaning, fill in missing values, drop duplicated data, change of datetine and remove repeated words to prepare the data for sentiment analysis and topic modeling.
## Methodology
For sentiment analysis, a pre-trained sentiment analysis model is utilized based on a deep learning architecture. This model is trained on a large corpus of text and is capable of classifying text into positive, negative, or neutral sentiment categories.

For topic modeling, the Latent Dirichlet Allocation (LDA) algorithm, a popular technique for discovering latent topics in a collection of documents is employed. The LDA model was trained on the preprocessed customer review data, allowing us to identify the main topics discussed by customers in their reviews.
## Results and Interpretation
The analysis revealed several interesting findings:

- Surprisingly, it is observed that for some of the products, they received negative reviews although having high ratings, these products have high ratings but low compound scores. This might due to the reason that some of the reviewers want their comments to be at the top of the comments section, so that it is more noticeable, assuming that the reviews are ranked based on the ratings obtained. 
- The main topics discussed by customers included product quality and pricing which are the crucial aspects for a customer to make purchase decision.
- Sentiment analysis assist in identifying specific aspects of the products or services that received positive or negative feedbacks, providing insights into areas of improvement or customer satisfactions.
  
The sentiment distribution and topic clusters are visualized through intuitive histogram, scatter plot and heatmap, which are included in the project repository.
## Business Application
Based on the analysis, we shall conclude that reviews of a product play a huge role in influencing customers' purchase decision and business owners' business decision.
- Positive reviews assist in customer acquisition and conversion while developing brand advocacy and products/services validation.
- On the other hand, negative reviews help the business owners to identify pain points of why the business is going downhill and to work on quality improvement in order to foster customer retention and for customer service enhancement.
## Method Used
- Webscraping - Selenium
- Sentiment Analysis - Vader & RoBERTa
- Topic Modelling - Latent Semantic Analysis(LSA), Latent Dirichlet Allocation (LDA) & BERTopic
- Deployment - Streamlit
## Streamlit
A Streamlit app is created to test the reviews.

[Streamlit](https://www.streamlit.io/)
