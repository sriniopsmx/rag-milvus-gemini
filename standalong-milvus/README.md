# This is a set of YAMLs for running milragv2 in k8s.

## Prerequisites
- milvus standalone is running in mil-ns namespace: https://artifacthub.io/packages/helm/milvus/milvus
- A collection with contents should have been created using MilvusBDUpdate.ipynb (model: entence-transformers/all-MiniLM-L6-v2)

## Instructions
- milrag-cm.yaml - env variables required in the script: ```k replace --force -f  milconfig-cm.yaml```
- ```k create secret generic milrag-google-api-key  --from-literal GOOGLE_API_KEY={YOUR-GOOGLE-API-KEY}```
- ```k delete cm milragpy; k create cm milragpy --from-file  MilvusRag.py``` # file name is important, don't change
- ```k replace --force -f milrag-dep.yaml```

Latest prompt-change (methods to confirm issue) has been done manually in the cluster.
