# Sentiment-analysis-project

## Project Overview
This project demonstrates sentiment analysis on the NLTK movie reviews dataset using machine learning techniques. The project includes data preprocessing, feature extraction using TF-IDF, and the implementation of two machine learning models: Multinomial Naive Bayes and Logistic Regression. The aim is to classify movie reviews as positive or negative based on their content.

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

## Requirements
The 'Requirements.txt' file lists all the Python packages required to run the project. Install these dependencies to avoid any compatibility issues.

## Results
1) The accuracy of the Multinomial Naive Bayes model: [0.785].
2) The accuracy of the Logistic Regression model: [0.795].
3) Detailed classification reports for both models are available in classification_report.txt.
4) Distribution of the sentiment categories is visualised in distribution_of_sentiment_categories.png.

## Conclusion
The sentiment analysis project successfully demonstrates the application of natural language processing and machine learning techniques to classify movie reviews as positive or negative. The Multinomial Naive Bayes and Logistic Regression models both performed well, with Logistic Regression slightly outperforming in terms of accuracy.

