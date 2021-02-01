#!/bin/sh

BUCKET="kfsconfig"

echo "\nCopy component specifications to Google Cloud Storage"
gsutil cp preprocess/component.yaml gs://${BUCKET}/componentz/preprocess/component.yaml
gsutil acl ch -u AllUsers:R gs://${BUCKET}/componentz/preprocess/component.yaml

echo "\nCopy component specifications to Google Cloud Storage"
gsutil cp training/component.yaml gs://${BUCKET}/componentz/training/component.yaml
gsutil acl ch -u AllUsers:R gs://${BUCKET}/componentz/training/component.yaml

echo "\nCopy component specifications to Google Cloud Storage"
gsutil cp testing/component.yaml gs://${BUCKET}/componentz/testing/component.yaml
gsutil acl ch -u AllUsers:R gs://${BUCKET}/componentz/testing/component.yaml
