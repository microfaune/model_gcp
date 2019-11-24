```
export GOOGLE_APPLICATION_CREDENTIALS="/Users/hadrien/microfaune-055a4872f9f6.json"
BUCKET_NAME="microfaune-hadrien"
REGION=europe-west1
MODEL_NAME='birdDetectionModel'
VERSION_NAME='v1'
```

Then run

```
gcloud ai-platform models create $MODEL_NAME \
  --regions $REGION
```


```
gsutil cp ./dist/deploy_bird_detection-0.1.tar.gz gs://$BUCKET_NAME/custom_predictions/deploy_bird_detection-0.1.tar.gz
```

```
gcloud components install beta

gcloud beta ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --runtime-version 1.13 \
  --python-version 3.5 \
  --origin gs://$BUCKET_NAME/custom_predictions/model/ \
  --package-uris gs://$BUCKET_NAME/custom_predictions/deploy_bird_detection-0.1.tar.gz \
  --prediction-class predictor.MyPredictor
```
