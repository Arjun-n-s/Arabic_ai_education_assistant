# Arabic AI Education Assistant

An AI-powered web application that generates Arabic image captions and evaluates students' Arabic written answers using Deep Learning, NLP, and Flask.

## Project Overview

This project consists of two modules:

### 1. Arabic Image Captioning
- Uses a pretrained **ResNet50 CNN** for image feature extraction.
- Uses an **LSTM with Attention** for English caption generation.
- Translates generated captions into Arabic.
- Displays the uploaded image with English and Arabic captions.

### 2. Automated Arabic Answer Evaluation
- Teachers enter questions and reference answers in Arabic.
- Students submit their answers.
- Uses **Sentence Transformers** to calculate semantic similarity.
- Generates a similarity-based score out of 10.

## Technologies Used

- Python
- TensorFlow and Keras
- ResNet50
- LSTM with Attention
- Sentence Transformers
- Flask
- HTML and CSS
- NumPy
- Pillow
- Deep Translator

## Dataset

**Flickr8k Dataset** — Used for image captioning model training and evaluation.

## Results

- Image Captioning BLEU-4 Score: **0.1698** (100 test images)
- Arabic Answer Evaluation: Semantic similarity-based scoring out of 10
- Both modules integrated into a Flask web application

## Project Structure

```text
Arabic-AI-Education-Assistant/
├── app.py
├── templates/
├── screenshots/
├── requirements.txt
├── README.md
└── .gitignore
```

The trained captioning model and tokenizer are required to run the application but are not included directly in this repository.

## How to Run

1. Clone the repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place `final_caption_model.keras` and `tokenizer.pkl` in the project folder.

4. Start the Flask application:

```bash
python app.py
```

5. Open the application:

   http://127.0.0.1:5000/

**Note:** This address works only when the Flask application is running locally. It is not a publicly deployed website.

## Future Improvements

- Improve caption generation accuracy.
- Train directly on Arabic image-caption datasets.
- Improve answer evaluation for incomplete and partially correct responses.
- Deploy the application online.
