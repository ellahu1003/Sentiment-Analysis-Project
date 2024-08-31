# Sentiment-analysis-project

## Project Overview
This project demonstrates sentiment analysis on the NLTK movie reviews dataset using machine learning techniques. The project includes data preprocessing, feature extraction using TF-IDF vectorisation, and the implementation of two machine learning models: Multinomial Naive Bayes and Logistic Regression. The aim is to classify movie reviews as positive or negative based on their content.

## Technologies Used
1) Pandas
2) NLTK
3) Scikit-Learn
4) Seaborn
5) Matplotlib

## Dataset
The dataset used in this project is the NLTK movie reviews dataset. It contains 2,000 movie reviews categorized into positive and negative sentiments.
1) Source: NLTK library
2) Categories: Positive, Negative
3) Number of Reviews: 2,000

## Project Structure
```markdown
sentiment-analysis/
│
├── data/
│ └── movie_reviews_dataset.csv
│
├── notebook/
│ └── sentiment_analysis.ipynb
│
├── results/
│ ├── distribution_of_sentiment_categories.png
│ └── classification_report.txt
│
└── requirements.txt
```
## Methodology
1) Data Collection:

   The dataset used for this project is the NLTK movie reviews dataset, containing 2,000 labeled movie reviews (positive or negative).

3) Data Preprocessing:
   1. Loading the Data: Using NLTK's built-in functions.
   2. Cleaning the Text Data: Removing stopwords, converting to lowercase, and removing punctuation.
   3. Tokenization: Converting text data into individual words.
   4. Dataframe Creation: Converting the cleaned data into a Pandas DataFrame and saving as a CSV file.
   
3) Exploratory Data Analysis (EDA):
   1. Checking for missing values and removing duplicates.
   2. Visualizing the distribution of sentiment categories.
   
4) Feature Extraction:

   Using TF-IDF vectorization to transform text data into numerical features.

6) Model Building and Training:
   1. Splitting the data into training and testing sets (80-20 split).
   2. Training a Multinomial Naive Bayes classifier and a Logistic Regression model on the TF-IDF features.
   
7) Model Evaluation:

   Calculating the accuracy score and generating classification reports for both models.

## Setup and Installations
1) Clone the repository:
    ```bash
    git clone https://github.com/ellahu1003/sentiment-analysis-project.git
    cd sentiment-analysis-project
    ```
2) Install the required libraries:
    ```bash
    pip install -r Requirements.txt
    ```
3) Run the Jupyter Notebook:
   ```bash
   jupyter notebook notebook/sentiment_analysis.ipynb
   ```

## Requirements
The 'Requirements.txt' file lists all the Python packages required to run the project. Install these dependencies to avoid any compatibility issues.

## Results
1) The accuracy of the Multinomial Naive Bayes model: [0.785].
2) The accuracy of the Logistic Regression model: [0.795].
3) Detailed classification reports for both models are available in classification_report.txt.
4) Distribution of the sentiment categories is visualised in distribution_of_sentiment_categories.png.

## Conclusion
The sentiment analysis project successfully demonstrates the application of natural language processing and machine learning techniques to classify movie reviews as positive or negative. The Multinomial Naive Bayes and Logistic Regression models both performed well, with Logistic Regression slightly outperforming in terms of accuracy.

