# Hausa-Yoruba Fake News Detection

Flask web application for detecting genuine or fake/misleading news in Hausa and Yoruba.

## Features
- Hausa and Yoruba selection
- Lightweight language validation before classification
- Fake-news prediction using TF-IDF + Logistic Regression
- Model confidence score
- Railway-compatible Flask deployment

## Railway start command
```text
gunicorn app:app
```

## Important
The language validation is a lightweight vocabulary-based check. It is not a separately trained language-identification model. The confidence shown by the website is the Logistic Regression model probability and should be described as model confidence, not proof that a news claim is true.
