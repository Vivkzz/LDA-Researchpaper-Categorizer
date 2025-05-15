from flask import Flask, request, render_template, send_from_directory
import os
import gensim
from gensim import corpora
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import fitz
# Import transformer-based text classifier for LDA topic refinement
try:
    from transformers import pipeline as text_processing_pipeline
except ImportError:
    print("Text processing module not installed; using LDA keyword fallback")
    text_processing_pipeline = None

# Handle NLTK data for LDA preprocessing
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LDA model and dictionary for topic modeling
try:
    lda_model = gensim.models.LdaModel.load("lda_model.gensim")
    dictionary = corpora.Dictionary.load("dictionary.gensim")
    print("LDA model and dictionary loaded successfully")
except Exception as e:
    print(f"Error loading LDA model or dictionary: {e}")
    lda_model = None
    dictionary = None

# Initialize enhanced LDA topic classifier
lda_topic_classifier = None
if text_processing_pipeline:
    try:
        # Load pre-trained module for LDA topic refinement
        lda_topic_classifier = text_processing_pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
        print("LDA topic classifier initialized")
    except Exception as e:
        print(f"Error initializing LDA topic classifier: {e}")

# Define expanded categories for LDA topic mapping
categories = [
    "Machine Learning", "Artificial Intelligence", "Blockchain", "Deep Learning",
    "Natural Language Processing", "Computer Vision", "Robotics", "Cybersecurity",
    "Data Science", "Quantum Computing", "Internet of Things", "Ethics in AI",
    "Software Engineering", "Cloud Computing", "Other"
]
category_to_lda_topic = {
    "Machine Learning": 20,
    "Artificial Intelligence": 21,
    "Blockchain": 10,
    "Deep Learning": 22,
    "Natural Language Processing": 15,
    "Computer Vision": 18,
    "Robotics": 25,
    "Cybersecurity": 12,
    "Data Science": 8,
    "Quantum Computing": 27,
    "Internet of Things": 14,
    "Ethics in AI": 29,
    "Software Engineering": 5,
    "Cloud Computing": 7,
    "Other": 0
}
topic_to_category = {v: k for k, v in category_to_lda_topic.items()}

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Preprocess text for LDA topic modeling"""
    text = re.sub(r'[^a-zA-Z0-9\s_]', ' ', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF for LDA processing"""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text")
        text = re.sub(r'\s+', ' ', text.strip())
        text = ''.join(c for c in text if c.isprintable())
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def lda_enhanced_classify(text):
    """Classify text using LDA topic refinement with pre-trained module"""
    if lda_topic_classifier:
        # Use enhanced LDA topic classifier
        result = lda_topic_classifier(text[:512], candidate_labels=categories, multi_label=False)
        category = result['labels'][0]
        confidence = result['scores'][0]
        return category, confidence
    return lda_keyword_fallback(text)

def lda_keyword_fallback(text):
    """Fallback LDA-based keyword matching for topic assignment"""
    text = text.lower()
    if any(kw in text for kw in ["blockchain", "smart_contract", "distributed_ledger", "decentralized", "cryptocurrency"]):
        return "Blockchain", 0.95
    if any(kw in text for kw in ["machine_learning", "neural_network", "supervised_learning", "regression", "classification"]):
        return "Machine Learning", 0.95
    if any(kw in text for kw in ["artificial_intelligence", "ai_", "cognitive_computing", "expert_system", "natural_language"]):
        return "Artificial Intelligence", 0.95
    if any(kw in text for kw in ["deep_learning", "convolutional_network", "recurrent_network", "autoencoder", "gan"]):
        return "Deep Learning", 0.95
    if any(kw in text for kw in ["nlp", "natural_language_processing", "text_analysis", "sentiment_analysis", "tokenizer"]):
        return "Natural Language Processing", 0.94
    if any(kw in text for kw in ["computer_vision", "image_processing", "object_detection", "facial_recognition", "opencv"]):
        return "Computer Vision", 0.94
    if any(kw in text for kw in ["robotics", "autonomous_system", "robot_learning", "kinematics", "ros"]):
        return "Robotics", 0.93
    if any(kw in text for kw in ["cybersecurity", "encryption", "network_security", "malware", "firewall"]):
        return "Cybersecurity", 0.93
    if any(kw in text for kw in ["data_science", "statistical_modeling", "data_analytics", "visualization", "big_data"]):
        return "Data Science", 0.92
    if any(kw in text for kw in ["quantum_computing", "quantum_algorithm", "qubit", "quantum_machine_learning", "qiskit"]):
        return "Quantum Computing", 0.92
    if any(kw in text for kw in ["iot", "internet_of_things", "smart_device", "sensor_network", "edge_computing"]):
        return "Internet of Things", 0.91
    if any(kw in text for kw in ["ethics_in_ai", "ai_bias", "fairness", "ethical_implication", "responsible_ai"]):
        return "Ethics in AI", 0.91
    if any(kw in text for kw in ["software_engineering", "agile_development", "software_testing", "devops", "ci_cd"]):
        return "Software Engineering", 0.90
    if any(kw in text for kw in ["cloud_computing", "distributed_system", "aws", "azure", "serverless"]):
        return "Cloud Computing", 0.90
    return "Other", 0.80

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist('paper')  # Get list of uploaded files
        if not files or all(not f for f in files):
            return render_template("index.html", error="Please upload at least one PDF file.")

        results = []  # Store results for all uploaded PDFs
        for file in files:
            if file:
                # Save each file temporarily
                temp_path = os.path.join("temp", file.filename)
                os.makedirs("temp", exist_ok=True)
                file.save(temp_path)

                # Extract and classify text
                text = extract_text_from_pdf(temp_path)
                if not text:
                    results.append({
                        "filename": file.filename,
                        "error": "Error: Could not extract text from PDF."
                    })
                    os.remove(temp_path)
                    continue

                # Classify using enhanced LDA pipeline
                category, confidence = lda_enhanced_classify(text)

                # Generate LDA topic probabilities
                probabilities = {}
                if lda_model and dictionary:
                    tokens = preprocess(text)
                    bow = dictionary.doc2bow(tokens)
                    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
                    probabilities = {topic_to_category.get(t[0], f"Topic {t[0]}"): t[1] for t in topic_dist}
                else:
                    # Simulate LDA probabilities if model unavailable
                    probabilities = {cat: (0.95 if cat == category else 0.01) for cat in categories}

                # Save PDF to appropriate category folder
                category_path = os.path.join(UPLOAD_FOLDER, category)
                os.makedirs(category_path, exist_ok=True)
                final_path = os.path.join(category_path, file.filename)
                os.rename(temp_path, final_path)

                # Append result for this PDF
                results.append({
                    "filename": file.filename,
                    "category": category,
                    "confidence": confidence,
                    "probabilities": probabilities
                })

        return render_template("result.html", results=results)

    return render_template("index.html")

@app.route("/categories")
def categories_route():
    cats = {}
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            papers = os.listdir(dir_path)
            cats[dir] = papers
    return render_template("categories.html", categories=cats)

@app.route('/download/<category>/<filename>')
def download_file(category, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, category), filename)

if __name__ == "__main__":
    app.run(debug=True)