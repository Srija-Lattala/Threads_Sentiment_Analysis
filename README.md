# Instagram Threads Sentiment Analysis

This project leverages machine learning and natural language processing (NLP) techniques to analyze the sentiment of user comments on Instagram Threads. By classifying comments as positive, negative, or neutral, the project aims to provide insights into user reactions and engagement patterns.

**Key Highlights:**

* **Data Extraction and Sentiment Analysis:** User comments are extracted from Play Store reviews and used to build a dataset for sentiment analysis. NLP techniques are applied to automatically classify comments as positive, negative, or neutral. This dataset is then adapted and applied to Instagram Threads comments for sentiment prediction.
* **Interactive Visualization:** An interactive web app is deployed on Streamlit, enabling users to explore sentiment trends visually. This facilitates a deeper understanding of user engagement and sentiment distribution.
* **Actionable Insights:** The project delivers actionable insights that can help in understanding user reactions, identifying potential issues, and fostering a more engaging and positive community.

## How it Works

1. **Data Collection and Preprocessing:** Data is initially extracted from Play Store reviews (using methods compliant with Google Play's terms of service). The collected text comments are then preprocessed using NLP techniques, including tokenization, stop word removal, and lemmatization. This data is stored in the `threads_comments.csv` file. This dataset is then adapted and applied to Instagram Threads comments for sentiment prediction.
2. **Model Training:** A machine learning model, such as a Support Vector Machine (SVM), is trained on the labeled dataset from `threads_comments.csv`, which was originally derived from Play Store reviews. The model learns patterns and relationships between words and sentiment to make accurate predictions. This process is handled by the `threads.py` script, which also creates the `svm_model.pkl` and `tfidf_vectorizer.pkl` files.
3. **Sentiment Prediction:** The trained model is deployed in the Streamlit web app. Users can input new Instagram Threads comments, and the model predicts their sentiment in real-time.
4. **Visualization and Insights:** The web app visualizes the sentiment trends using charts and graphs, providing users with an overview of the sentiment distribution across different comments or time periods. This allows for deeper analysis and understanding of user engagement patterns.

## Technologies Used

* Python
* Streamlit (for creating the interactive web app)
* Scikit-learn (for machine learning models and tools)
* NLTK (for text preprocessing)
* SpaCy (for natural language processing)
* Pickle (for saving and loading the model and vectorizer)

## Setup and Usage

1. Clone this repository: `git clone https://github.com/your-username/threads_sentiment_analysis.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app locally: `streamlit run deployment.py`

## Live Demo

**Check out the live demo of the app here:** [https://threads-sentiment-analyzer.streamlit.app/]  

## Files

* `deployment.py`: Main Streamlit app script.
* `threads.py`: Script for creating the SVM model and TF-IDF vectorizer, generating the `svm_model.pkl` and `tfidf_vectorizer.pkl` files.
* `svm_model.pkl`: Trained SVM model.
* `tfidf_vectorizer.pkl`: TF-IDF vectorizer.
* `threads_comments.csv`: Dataset containing Instagram Threads comments and sentiment labels used for model training (originally derived from Play Store reviews).
* `requirements.txt`: List of project dependencies and Python version to use.


## Contributing

Contributions are welcome! Feel free to open issues or pull requests for bug fixes, feature enhancements, or improvements to the model.

## License

This project is licensed under the [MIT License](LICENSE).
