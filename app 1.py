from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Initialize the Flask application
app = Flask(__name__)

# Load the dataset
df = pd.read_csv("./first_350_rows.csv")

# Replace NaN values in 'Skills' column with an empty string
df['Skills'].fillna('', inplace=True)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stop words
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit_form():
    global df  # Ensure df is accessible in this function

    if request.method == 'POST':
        # Process form data
        user_data = request.json
        user_data = {k: v if isinstance(v, list) else [v] for k, v in user_data.items()}
        
        # Combine user information into a single string
        user_profile = " ".join(user_data.get("skills", [])) + " " + user_data.get("learning_goals", [""])[0]
        
        # Preprocess the user profile
        user_profile = preprocess_text(user_profile)
        
        # Preprocess the 'Skills' column in the dataframe
        df['Skills'] = df['Skills'].apply(preprocess_text)
        
        # Combine multiple relevant columns into a single feature
        df['Combined_Features'] = df['Title'] + ' ' + df['Category'] + ' ' + df['Sub-Category'] + ' ' + df['Skills']
        
        # Remove courses that the user has already completed based on certificates
        for certificate in user_data.get("certificates", []):
            df = df[~df['Skills'].str.contains(certificate.lower())]
        
        # Prepare data for model training
        X = df['Combined_Features']
        y = df['Category']  # Assuming 'Category' is the target variable for classification
        
        # Convert text data into TF-IDF features
        tfidf = TfidfVectorizer(stop_words='english')
        X_tfidf = tfidf.fit_transform(X)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        best_dt_model = grid_search.best_estimator_
        best_dt_model.fit(X_train, y_train)
        
        # Predict the user's course recommendations
        user_profile_tfidf = tfidf.transform([user_profile])
        predicted_category = best_dt_model.predict(user_profile_tfidf)[0]
        
        # Filter courses based on the predicted category
        recommended_courses = df[df['Category'] == predicted_category]
        
        # Filter courses based on job title
        job_title_courses = df[df['Title'].str.contains(user_data.get("job_title", [""])[0], case=False)].head(2)
        
        # Filter courses based on learning categories and learning goals
        learning_courses = df[(df['Category'] == predicted_category) & 
                              (df['Skills'].str.contains(user_data.get("learning_categories", [""])[0].lower())) & 
                              (df['Skills'].str.contains(preprocess_text(user_data.get("learning_goals", [""])[0])))]
        learning_courses = learning_courses.head(3)
        
        # Combine all recommended courses
        recommended_courses = pd.concat([job_title_courses, learning_courses], ignore_index=True)
        
        # Convert recommended courses to dictionary format
        recommended_courses = recommended_courses[['Title', 'URL', 'Category', 'Sub-Category', 'Skills', 'Rating', 'Organization']].to_dict('records')
        
        return jsonify(recommended_courses=recommended_courses)

if __name__ == '__main__':
    app.run(debug=True)
