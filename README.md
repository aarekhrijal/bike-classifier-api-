# Bike Brand Classifier API

A REST API that classifies bike images into Honda, Yamaha, or KTM using ResNet34 and fastai.

## Live API
[Railway URL will go here]

## API Endpoints

- `GET /` — Health check
- `POST /predict` — Classify a bike image

## Example Response
```json
{
  "predicted_class": "honda bike",
  "confidence": 0.7545
}
```

## Tech Stack
- FastAPI
- fastai / ResNet34
- PyTorch
- Railway (deployment)