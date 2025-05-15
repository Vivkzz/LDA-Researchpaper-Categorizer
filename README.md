# LDA Researchpaper Categorizer

This project is a Flask web application for automatic categorization of research papers (PDFs) using LDA topic modeling and transformer-based (zero-shot) classification. It allows users to upload research papers, automatically classifies them into relevant categories, and organizes them for easy browsing and download.

## Features
- Upload one or multiple research papers (PDF format)
- Automatic extraction of text from PDFs
- Topic classification using:
  - Pre-trained transformer (zero-shot classification with Facebook BART)
  - LDA topic modeling (Gensim)
  - Keyword-based fallback if models are unavailable
- Categories include Machine Learning, Artificial Intelligence, Blockchain, Deep Learning, NLP, Computer Vision, Robotics, Cybersecurity, Data Science, Quantum Computing, IoT, Ethics in AI, Software Engineering, Cloud Computing, and Other
- Categorized storage and download of uploaded papers
- Simple, modern UI (Tailwind CSS)

## Requirements
- Python 3.8+
- pip

### Python Packages
- Flask
- gensim
- nltk
- PyMuPDF (fitz)
- transformers
- torch
- numpy

Install all requirements:
```bash
pip install flask gensim nltk pymupdf transformers torch numpy
```

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Vivkzz/LDA-Researchpaper-Categorizer.git
   cd LDA-Researchpaper-Categorizer
   ```
2. Download NLTK data (first run will do this automatically, or run manually):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
3. Place your pre-trained LDA model and dictionary files (`lda_model.gensim`, `dictionary.gensim`) in the project root, or train your own.

## Usage
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
3. Upload PDF(s) and view categorized results.
4. Browse all categorized papers at `/categories`.

## Project Structure
- `app.py` - Main Flask application
- `templates/` - HTML templates (Jinja2)
- `static/uploads/` - Uploaded and categorized PDFs
- `lda_model.gensim`, `dictionary.gensim` - LDA model files

## Notes
- If the transformer model or LDA model is not available, the app falls back to keyword-based classification.
- For production, use a WSGI server (e.g., gunicorn) and secure file uploads.

## License
MIT License

## Author
[Your Name] (Vivkzz)
