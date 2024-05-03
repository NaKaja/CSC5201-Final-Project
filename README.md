# CSC5201 Spring 2024 Final Project
## Description
This is a fairly simple microservices application which serves as a wrapper around a transcript summarizer model. 
This model comes out of a Senior Design project from a year ago. It is a BART model fine-tuned on a dataset of Khan
Academy lectures which aims to summarize chunks of a lecture transcript into short descriptions that are similar to the
'About' section under any Khan Academy lecture video.  

This application aims to have 4 services:  
- **App Service**: The frontend which users interact with to login, logout, and submit text forms to the model
- **Model Service**: A wrapper for the HuggingFace model which takes in text and prediction parameters
- **MINIO Service** (not implemented): An object store service where user text files can be stored and retrieved later
- **Prometheus Service**: A monitoring services which periodically queries all other services for endpoint metrics like count and latency

<img src='./Service Diagram.png'>

## Installation
This application is orchestrated through docker compose so installation is very simple. On a machine with Git and Docker installed:  

1. Clone the repository: `git clone https://github.com/NaKaja/CSC5201-Final-Project`
2. cd into the top project directory (with the docker-compose.yml): `cd CSC5201-Final-Project`
3. Start up the docker compose application: `docker-compose up`

*Note*: As the model is somewhat large, it is recommended to run this on a machine with a GPU. 
If running directly on a Windows machine, make sure your Docker Desktop is through the WSL2 backend as that is 
currently the only way to have GPU support.
