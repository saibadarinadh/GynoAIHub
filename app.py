from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pickle
import fitz  # PyMuPDF for PDF text extraction
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO


# Load models and vectorizer
with open('./model/model1.pkl', 'rb') as model_file:
    models = pickle.load(model_file)
with open('./model/tfid1.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load Label Encoders
le_targets = {
    'Disorder': LabelEncoder(),
    'Treatment Recommendation': LabelEncoder(),
    'Precautions': LabelEncoder(),
    'Food Intake Recommendation': LabelEncoder(),
    'Foods to Avoid': LabelEncoder(),
    'Duration of Symptoms': LabelEncoder(),
    'Lifestyle Recommendations': LabelEncoder(),
}


# Fit Label Encoders
for target in le_targets.keys():
    le_targets[target].fit(pd.read_csv('./data_set/gynecological_conditions.csv')[target])

def predict_outputs(symptoms):
    input_vectorized = vectorizer.transform([symptoms])
    predictions = {}
    for target in models.keys():
        predicted_encoded = models[target].predict(input_vectorized)
        predictions[target] = le_targets[target].inverse_transform(predicted_encoded)[0]
    return predictions

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page in pdf_document:
            text += page.get_text()
    return text

# Function to send email
def send_email(receiver_email, predictions):
    sender_email = "yasaswienuga@gmail.com"  # Replace with your email address
    sender_password = "zdkxcxjwyppksxij"  # Replace with your email password

    # Create the email content
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Prediction Results Based on Your Symptoms"

    body = (
        f"Based on your symptoms, here is the information:\n"
        f"- Disorder: {predictions['Disorder']}\n"
        f"- Treatment Recommendation: {predictions['Treatment Recommendation']}\n"
        f"- Precautions: {predictions['Precautions']}\n"
        f"- Food Intake Recommendation: {predictions['Food Intake Recommendation']}\n"
        f"- Foods to Avoid: {predictions['Foods to Avoid']}\n"
        f"- Duration of Symptoms: {predictions['Duration of Symptoms']}\n"
        f"- Lifestyle Recommendations: {predictions['Lifestyle Recommendations']}"
    )

    message.attach(MIMEText(body, "plain"))

    try:
        # Set up the server
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Example for Gmail SMTP
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'

class MedicalPredictor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.age_encoder = LabelEncoder()
        self.model = None
        self.data = None
        
        # Load the model and encoders
        try:
            with open('./backend/test-1/medical_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['classifier']
                self.tfidf = model_data['tfidf']
                self.age_encoder = model_data['age_encoder']
                self.data = model_data['data']
                
                # Ensure age encoder is fitted with correct values
                self.age_encoder.fit(['under 13 years', '13-45 years', 'over 45 years'])
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def find_closest_age_group(self, age_group):
        """Find the closest matching age group from the available categories."""
        standard_groups = ['under 13 years', '13-45 years', 'over 45 years']
        
        # If input is a number, convert to appropriate group
        if isinstance(age_group, (int, float)):
            if age_group < 13:
                return "under 13 years"
            elif 13 <= age_group <= 45:
                return "13-45 years"
            else:
                return "over 45 years"
                
        # If it's a string, find the closest match
        age_group = age_group.lower()
        for group in standard_groups:
            if group.lower() in age_group or age_group in group.lower():
                return group
                
        # Default fallback
        return "13-45 years"

    def predict_condition(self, symptoms, age_group):
        try:
            # Preprocess symptoms
            symptoms_vector = self.tfidf.transform([symptoms])
            
            # Process age group
            standardized_age_group = self.find_closest_age_group(age_group)
            age_encoded = self.age_encoder.transform([standardized_age_group])
            
            # Combine features and predict
            X_pred = np.hstack((symptoms_vector.toarray(), age_encoded.reshape(-1, 1)))
            predicted_condition = self.model.predict(X_pred)[0]
            
            # Get full condition information
            condition_info = self.data[self.data['Disorder/Condition Name'] == predicted_condition].iloc[0]
            
            return {
                'Predicted Condition': condition_info['Disorder/Condition Name'],
                'Category': condition_info['Category'],
                'Typical Symptoms': condition_info['Symptoms'],
                'Diagnosis Tests': condition_info['Diagnosis Tests'],
                'Treatment Recommendations': condition_info['Treatment Recommendations'],
                'Medications': condition_info['Medications'],
                'Therapies/Procedures': condition_info['Therapies/Procedures'],
                'Food Recommendations': condition_info['Food Intake Recommendations'],
                'Foods to Avoid': condition_info['Foods to Avoid'],
                'Lifestyle Recommendations': condition_info['Lifestyle Recommendations'],
                'Duration': condition_info.get('Duration of Symptoms', 'Not specified'),
                'Recovery Time': condition_info.get('Recovery Time', 'Varies by individual'),
                'Prognosis': condition_info['Recovery Outlook/Prognosis']
            }
        except Exception as e:
            print(f"Prediction error details: {str(e)}")
            return {'error': f"Prediction error: {str(e)}"}

    def answer_question(self, question):
        """Improved method to answer specific questions about medical conditions"""
        try:
            question = question.lower()
            
            # Extract condition name from common variations
            condition = None
            for idx, row in self.data.iterrows():
                # Check for exact condition name or common variations
                condition_variations = [
                    row['Disorder/Condition Name'].lower(),
                    row['Disorder/Condition Name'].lower().replace(' syndrome', ''),
                    row['Disorder/Condition Name'].lower().replace(' disorder', ''),
                    row['Disorder/Condition Name'].lower().replace('syndrome', ''),
                    row['Disorder/Condition Name'].lower().replace('disorder', '')
                ]
                
                if any(var in question for var in condition_variations):
                    condition = row
                    break
            
            if condition is None:
                return "I couldn't identify a specific condition in your question. Please mention the condition name clearly in your question."
            
            # Identify question type and provide appropriate response
            if any(word in question for word in ['test', 'diagnos', 'check']):
                return f"Diagnostic tests for {condition['Disorder/Condition Name']}:\n{condition['Diagnosis Tests']}"
                
            elif any(word in question for word in ['symptom', 'sign', 'indication']):
                return f"Common symptoms of {condition['Disorder/Condition Name']}:\n{condition['Symptoms']}"
                
            elif any(word in question for word in ['treat', 'therapy', 'cure', 'medicine', 'medication']):
                return (f"Treatment information for {condition['Disorder/Condition Name']}:\n\n"
                       f"Medications:\n{condition['Medications']}\n\n"
                       f"Treatment Recommendations:\n{condition['Treatment Recommendations']}\n\n"
                       f"Therapies/Procedures:\n{condition['Therapies/Procedures']}")
                
            elif any(word in question for word in ['food', 'eat', 'diet', 'nutrition']):
                return (f"Dietary recommendations for {condition['Disorder/Condition Name']}:\n\n"
                       f"Recommended Foods:\n{condition['Food Intake Recommendations']}\n\n"
                       f"Foods to Avoid:\n{condition['Foods to Avoid']}")
                
            elif any(word in question for word in ['time', 'duration', 'long', 'recovery']):
                return (f"Duration and recovery information for {condition['Disorder/Condition Name']}:\n\n"
                       f"Duration of Symptoms: {condition.get('Duration of Symptoms', 'Not specified')}\n"
                       f"Recovery Time: {condition.get('Recovery Time', 'Varies by individual')}\n"
                       f"Prognosis: {condition['Recovery Outlook/Prognosis']}")
                
            elif any(word in question for word in ['lifestyle', 'activity', 'exercise', 'habit']):
                return f"Lifestyle recommendations for {condition['Disorder/Condition Name']}:\n{condition['Lifestyle Recommendations']}"
                
            else:
                # General information about the condition
                return (f"General information about {condition['Disorder/Condition Name']}:\n\n"
                       f"Category: {condition['Category']}\n\n"
                       f"Common Symptoms:\n{condition['Symptoms']}\n\n"
                       f"Treatment Overview:\n{condition['Treatment Recommendations']}")
                
        except Exception as e:
            print(f"Error in answer_question: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try rephrasing your question or ask about a different aspect of the condition."
# Initialize predictor
predictor = MedicalPredictor()


# Database setup
DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL
                          )''')
        conn.commit()



# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Sign-up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        try:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists. Try a different username."
    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            if result and check_password_hash(result[0], password):
                session['user'] = username
                return redirect(url_for('dashboard'))
            else:
                return "Invalid credentials, please try again."
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# Dashboard route
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

# Chatbot page
@app.route('/chatbot')
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html')


@app.route('/faq')
def faq_page():
    return render_template('FAQ Page.html')

@app.route('/Articles')
def Articles():
    return render_template('Health Articles Page.html')

# Chatbot text response route
@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_input = data.get('message', '')

    try:
        # Check if the input is a question
        question_indicators = ['what', 'how', 'when', 'where', 'why', 'can', 'does', 'is', 'are', '?']
        is_question = any(indicator in user_input.lower() for indicator in question_indicators)

        if is_question:
            # Handle question logic from the new code
            response = predictor.answer_question(user_input)
            return jsonify({"response": response})

        # Handle symptom input logic using the old code
        predictions = predict_outputs(user_input)
        response_text = (
            f"Based on your symptoms, here is the information:\n"
            f"- Disorder: {predictions['Disorder']}\n"
            f"- Treatment Recommendation: {predictions['Treatment Recommendation']}\n"
            f"- Precautions: {predictions['Precautions']}\n"
            f"- Food Intake Recommendation: {predictions['Food Intake Recommendation']}\n"
            f"- Foods to Avoid: {predictions['Foods to Avoid']}\n"
            f"- Duration of Symptoms: {predictions['Duration of Symptoms']}\n"
            f"- Lifestyle Recommendations: {predictions['Lifestyle Recommendations']}"
        )
        return jsonify({"response": response_text})

    except Exception as e:
        print(f"Error in get_response: {str(e)}")  # For debugging
        return jsonify({
            "response": """An error occurred. Please try one of these formats:

1. For symptoms analysis:
name: [your name], age: [your age], weight: [your weight], gender: [your gender], [symptom1], [symptom2], ...

2. For questions:
Simply ask about a specific condition, e.g., "What are the symptoms of PCOS?"

Error details: """ + str(e)
        })

@app.route('/pdf_upload')
def pdf_upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('pdf_upload.html')
# Chatbot PDF upload response route

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"response": "No file uploaded"}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"response": "No file selected"}), 400

    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(pdf_path)

    extracted_text = extract_text_from_pdf(pdf_path)
    predictions = predict_outputs(extracted_text)
    response_text = (
        f"Based on your symptoms, here is the information:\n"
        f"- Disorder: {predictions['Disorder']}\n"
        f"- Treatment Recommendation: {predictions['Treatment Recommendation']}\n"
        f"- Precautions: {predictions['Precautions']}\n"
        f"- Food Intake Recommendation: {predictions['Food Intake Recommendation']}\n"
        f"- Foods to Avoid: {predictions['Foods to Avoid']}\n"
        f"- Duration of Symptoms: {predictions['Duration of Symptoms']}\n"
        f"- Lifestyle Recommendations: {predictions['Lifestyle Recommendations']}"
    )
    receiver_email = "yasaswi003@gmail.com"  # Replace with the user's email address
    send_email(receiver_email, predictions)
    return jsonify({"response": response_text})



@app.route('/pcos_detection', methods=['GET', 'POST'])
def pcos_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"response": "No file uploaded"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"response": "No file selected"}), 400

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # New Model: Check if the Image is Related to the Dataset
        vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        new_model_path = r'./image based/PCOS_Detection/if or not/saved_related_model.pkl'
        with open(new_model_path, 'rb') as file:
            dataset_features = pickle.load(file)

        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        features = vgg_model.predict(img_array).flatten()

        from sklearn.metrics.pairwise import cosine_similarity
        max_similarity = np.max(cosine_similarity([features], dataset_features))

        if max_similarity < 0.95:
            return render_template('pcos_detection.html', prediction=f"Not related to the dataset (Similarity: {max_similarity:.2f})")

        # Existing PCOS Detection Model
        pcos_model = tf.keras.models.load_model('./image based/PCOS_Detection/final_model.keras')

        img = load_img(image_path, target_size=(128, 128))  # Resize to match training input
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        prediction = pcos_model.predict(img_array)
        predicted_class = (prediction[0] > 0.5).astype("int32")

        if predicted_class == 1:
            prediction = "The image is not infected with PCOS."
        else:
            prediction = "The image is infected with PCOS."

        return render_template('pcos_detection.html', prediction=prediction)
    
    return render_template('pcos_detection.html')
@app.route('/cervical_cancer', methods=['GET', 'POST'])
def cervical_cancer():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('cervical_cancer.html', 
                                predicted_class="No file uploaded",
                                confidence_scores=[])
        
        file = request.files['image']
        if file.filename == '':
            return render_template('cervical_cancer.html', 
                                predicted_class="No file selected",
                                confidence_scores=[])

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # First: Check if the image is related to the dataset using VGG16
        vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        new_model_path = r'./image based/Cervical Cancer/if or not/saved_related_model.pkl'
        
        with open(new_model_path, 'rb') as file:
            dataset_features = pickle.load(file)

        # Preprocess image for VGG16
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        features = vgg_model.predict(img_array).flatten()

        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        max_similarity = np.max(cosine_similarity([features], dataset_features))

        print(f"Similarity score: {max_similarity}")  # Debug print

        # If image is not related to the dataset, return early
        if max_similarity < 0.95:
            return render_template('cervical_cancer.html', 
                                predicted_class=f"Image is not related to cervical cancer dataset (Similarity: {max_similarity:.2f})",
                                confidence_scores=[0, 0, 0, 0, 0, 0])  # Empty scores for unrelated image

        try:
            # If image is related, proceed with cervical cancer classification
            model = tf.keras.models.load_model('./image based/Cervical Cancer/best_model.keras')

            # Preprocess the image for classification
            def preprocess_image(image_path):
                img = load_img(image_path, target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                return img_array

            def predict_image(model, image_path):
                processed_image = preprocess_image(image_path)
                predictions = model.predict(processed_image)
                predicted_class = np.argmax(predictions, axis=1)[0]
                class_labels = {0: 'Dyskeratotic', 1: 'Koilocytotic', 2: 'Metaplastic', 
                              3: 'Normal', 4: 'Parabasal', 5: 'Superficial-Intermediate'}
                return class_labels[predicted_class], predictions[0]

            predicted_class, confidence_scores = predict_image(model, image_path)
            
            # Format the result message
            result_message = f"{predicted_class} (Image is related, Similarity: {max_similarity:.2f})"
            
            return render_template('cervical_cancer.html', 
                                predicted_class=result_message,
                                confidence_scores=confidence_scores.tolist())

        except Exception as e:
            print(f"Error processing image: {str(e)}")  # Debug print
            return render_template('cervical_cancer.html', 
                                predicted_class=f"Error processing image: {str(e)}",
                                confidence_scores=[0, 0, 0, 0, 0, 0])

    return render_template('cervical_cancer.html')
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
