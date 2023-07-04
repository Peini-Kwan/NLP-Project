# Sentiment Analysis and Topic Modeling of The North Face Store on Amazon E-commerce
## Project Overview
This project aims to perform sentiment analysis and topic modeling on customer reviews of The North Face store on Amazon. By analyzing customers' sentiment and identifying key topics, the intention of this project is to to gain valuable insights into the opinions and preferences of customers regarding The North Face products available on the e-commerce platform.
## Data Collection
A dataset of customer reviews is collected from The North Face store, under Men's Section on Amazon using web scraping techniques. The dataset consists of textual reviews along with corresponding ratings and other metadata. Before analysis, data preprocessing steps such as text cleaning, fill in missing values, drop duplicated data, change of datetine and remove repeated words to prepare the data for sentiment analysis and topic modeling.
## Methodology
For sentiment analysis, a pre-trained sentiment analysis model is utilized based on a deep learning architecture. This model is trained on a large corpus of text and is capable of classifying text into positive, negative, or neutral sentiment categories.

For topic modeling, the Latent Dirichlet Allocation (LDA) algorithm, a popular technique for discovering latent topics in a collection of documents is employed. The LDA model was trained on the preprocessed customer review data, allowing us to identify the main topics discussed by customers in their reviews.
## Results and Interpretation
Our analysis revealed several interesting findings:

- Overall, customer sentiment towards The North Face products on Amazon was predominantly positive, with a majority of reviews expressing satisfaction and appreciation.
- The main topics discussed by customers included product quality and pricing which are the crucial aspects for a customer to make purchase decision.
- Sentiment analysis helped identify specific aspects of the products or services that received positive or negative feedback, providing insights into areas of improvement or customer satisfaction.
  
We visualized the sentiment distribution and topic clusters through intuitive charts and word clouds, which are included in the project repository.
## Streamlit
A Streamlit app is created to test the reviews.
[Streamlit](https://www.streamlit.io/)



