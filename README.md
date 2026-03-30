# Zero-Shot Emotion Detection System 🎭

##  Overview

This project uses **RoBERTa (MNLI)** for detecting emotions from text using **Zero-Shot Learning**.

##  Features

* Zero-shot classification (no training required)
* Multi-label emotion detection
* Batch processing
* CLI-based input
* JSON output support

##  Technologies Used

* Python
* Hugging Face Transformers
* PyTorch

## How to Run

```bash
pip install -r requirements.txt
python zero_shot_emotion_detector.py --text "I am happy"
```

##  Example Output

```
Text: I am happy
joy → 0.92
Top: joy (0.92)
```

##  Author

Vaaridhi Das
