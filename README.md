# Women's Health Portal

This project is an end-to-end application built with Flask for diagnosing and predicting various women's health conditions. It integrates machine learning models, deep learning models, PDF generation, and image-based detection tools to assist in medical reporting and decision support.

## Project Structure

- **Root Files**
  - [app.py](app.py): The main Flask application that handles routing, uploads, predictions, and interacts with the machine learning models.
  - [pdf_generate.py](pdf_generate.py): A standalone script to generate medical reports using ReportLab.
  - **Data Files**
    - `1.pdf`, `2.pdf`: Example PDF reports generated or used within the system.
    - `users.db`: SQLite database to manage user information.

- **Backend**
  - [backend/gynaecology.ipynb](backend/gynaecology.ipynb): Notebook for training and predicting gynecological conditions. It includes functions for text extraction from PDFs and symptom-based prediction.
  - [backend/slr.ipynb](backend/slr.ipynb) and [backend/slr copy.ipynb](backend/slr%20copy.ipynb): Notebooks containing similar prediction logic using ensemble models.
  - Other backend notebooks such as [backend/test-1/one.ipynb](backend/test-1/one.ipynb) and [backend/test-2.ipynb](backend/test-2.ipynb) are used for testing and evaluating model performance.

- **Data Set**
  - [data_set/gynecological_conditions.csv](data_set/gynecological_conditions.csv): CSV data file used for training the machine learning models.
  - Other CSV files include patient and condition-specific data.

- **Models**
  - [model/model1.pkl](model/model1.pkl): Trained model file for condition prediction.
  - [model/tfid1.pkl](model/tfid1.pkl): Trained TF-IDF vectorizer used for text processing.

- **Templates**
  - HTML files in the [templates](templates/) folder define user interfaces for:
    - Cervical cancer detection ([cervical_cancer.html](templates/cervical_cancer.html))
    - PCOS detection ([pcos_detection.html](templates/pcos_detection.html))
    - PDF Upload for detection ([pdf_upload.html](templates/pdf_upload.html))
    - Dashboard, Chatbot, and other pages.
  - [templates/pdf_generate.py](templates/pdf_generate.py) provides additional PDF generation logic tied to web functionality.

- **Static**
  - [static/style.css](static/style.css): Contains styling rules to ensure a modern and responsive UI.

- **Uploads**
  - The `uploads/` folder holds files uploaded by users for processing (such as PDFs for symptom extraction).

## Key Features

- **Medical Prediction Engine**  
  Uses machine learning and deep learning to predict disorders, treatment recommendations, precautions, food intake suggestions, and more based on extracted symptoms from PDFs or images. See the prediction logic in [app.py](app.py) and [backend/gynaecology.ipynb](backend/gynaecology.ipynb).

- **PDF Text Extraction and Report Generation**  
  Leverages PyMuPDF (fitz) for extracting text from PDFs and ReportLab for generating clean, formatted reports. Refer to the extraction functions in [backend/gynaecology.ipynb](backend/gynaecology.ipynb) and the report generation code in [pdf_generate.py](pdf_generate.py).

- **Image-Based Detection**  
  Provides an interface for image uploads and uses deep learning models for detecting conditions like PCOS. The UI is handled by HTML templates such as [pcos_detection.html](templates/pcos_detection.html).

- **User Management**  
  Simple user authentication and session management using Flask and SQLite in [app.py](app.py).

## Setup and Usage

1. **Install Dependencies**  
   Ensure you have Python installed and install the required packages:
   ```sh
   pip install -r requirements.txt
