#!/usr/bin/env python3
"""

ZERO-SHOT EMOTION DETECTION SYSTEM USING ROBERTA DATA MODEL (FINAL YEAR PROJECT)


Project Features:
 Zero-shot emotion classification using RoBERTa (MNLI)
 Supports CLI, File Input, and Batch Processing
 Multi-label classification
 Confidence threshold filtering
 JSON result export
 Logging system for debugging
 Modular & scalable architecture

Author: Vaaridhi Das

"""


# IMPORTS

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


# THIRD-PARTY IMPORTS

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except Exception as e:
    raise ImportError("Install transformers: pip install transformers")

try:
    import torch
    HAS_TORCH = True
except:
    HAS_TORCH = False
    torch = None



# CONFIGURATION CLASS

class Config:
    """Configuration settings for the project"""

    MODEL_NAME = "roberta-large-mnli"
    DEFAULT_LABELS = [
        "joy", "happiness", "excitement",
        "sadness", "disappointment",
        "anger", "frustration",
        "fear", "anxiety",
        "surprise", "shock",
        "disgust",
        "love", "affection",
        "neutral", "calm"
    ]

    LOG_FILE = "emotion_detector.log"
    OUTPUT_FILE = "results.json"
    BATCH_SIZE = 16



# LOGGER SETUP

def setup_logger():
    logging.basicConfig(
        filename=Config.LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )



# UTILITY FUNCTIONS

class FileHandler:
    """Handles file reading and writing"""

    @staticmethod
    def read_text_file(filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def save_json(data: List[Dict[str, Any]], filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)



# CORE MODEL CLASS

class EmotionDetector:
    """
    Core class for emotion detection using zero-shot classification.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = self._detect_device()
        self.classifier = self._load_model()

    def _detect_device(self) -> int:
        """Detect GPU or CPU"""
        if HAS_TORCH and torch and torch.cuda.is_available():
            logging.info("Using GPU")
            return 0
        logging.info("Using CPU")
        return -1

    def _load_model(self):
        """Load HuggingFace model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(self.config.MODEL_NAME)

            return pipeline(
                "zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )
        except Exception as e:
            logging.error("Model loading failed")
            raise RuntimeError(f"Error loading model: {e}")

    def predict(
        self,
        texts: Union[str, List[str]],
        labels: Optional[List[str]] = None,
        multi_label: bool = False,
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        if isinstance(texts, str):
            texts = [texts]

        if labels is None:
            labels = self.config.DEFAULT_LABELS

        results = []

        for i in range(0, len(texts), self.config.BATCH_SIZE):
            batch = texts[i:i + self.config.BATCH_SIZE]

            outputs = self.classifier(
                batch,
                labels,
                multi_label=multi_label,
                hypothesis_template="This text expresses {}."
            )

            if isinstance(outputs, dict):
                outputs = [outputs]

            for text, output in zip(batch, outputs):
                pairs = list(zip(output["labels"], output["scores"]))

                # Apply threshold
                pairs = [(l, s) for l, s in pairs if s >= threshold]

                # Top K
                pairs = pairs[:top_k]

                results.append({
                    "text": text,
                    "predictions": [
                        {"label": l, "score": round(s, 4)} for l, s in pairs
                    ],
                    "top_label": pairs[0][0] if pairs else None,
                    "top_score": round(pairs[0][1], 4) if pairs else None
                })

        return results



# EVALUATION MODULE

class Evaluator:
    """
    Basic evaluation utilities (for demonstration)
    """

    @staticmethod
    def print_summary(results: List[Dict[str, Any]]):
        print("\n========== SUMMARY ==========")
        for r in results:
            print(f"Text: {r['text']}")
            print(f"Top Emotion: {r['top_label']} ({r['top_score']})\n")



# DEMO DATA 

def demo_data() -> List[str]:
    return [
        "I just got placed in my dream company!",
        "I feel lonely and sad.",
        "This is the worst day ever.",
        "I am very excited for the trip!",
        "I am nervous about my exam.",
        "I love my parents so much.",
        "Nothing special happened today.",
        "That movie twist shocked me!",
        "I am angry at my friend.",
        "I feel calm and peaceful."
    ]



# CLI INTERFACE

def run_cli():
    parser = argparse.ArgumentParser(description="Emotion Detection System")

    parser.add_argument("--text", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--multi_label", action="store_true")
    parser.add_argument("--save", action="store_true")

    args, _ = parser.parse_known_args()

    config = Config()
    detector = EmotionDetector(config)

    inputs = []

    if args.text:
        inputs.append(args.text)

    if args.file:
        inputs.extend(FileHandler.read_text_file(args.file))

    if not inputs:
        print("No input provided. Running demo...")
        inputs = demo_data()

    results = detector.predict(
        inputs,
        multi_label=args.multi_label,
        top_k=args.top_k,
        threshold=args.threshold
    )

    # Print Results
    for r in results:
        print("\nText:", r["text"])
        for p in r["predictions"]:
            print(f"  {p['label']} → {p['score']}")
        print(f"Top: {r['top_label']} ({r['top_score']})")

    # Save if required
    if args.save:
        FileHandler.save_json(results, config.OUTPUT_FILE)
        print(f"\nSaved to {config.OUTPUT_FILE}")

    Evaluator.print_summary(results)



# MAIN FUNCTION

def main():
    setup_logger()
    logging.info("Program Started")
    run_cli()
    logging.info("Program Finished")


if __name__ == "__main__":
    main()


