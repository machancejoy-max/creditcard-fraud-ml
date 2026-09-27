This project is my end‑to‑end machine learning system for detecting credit card fraud. I built it to learn how real ML pipelines work from start to finish — from exploring the data, to training the model, to deploying it in a container.

The dataset needed a lot of cleaning and balancing because fraud cases were extremely rare. I used scaling and oversampling to help the model learn better. For the model itself, I chose XGBoost because it performs well on tabular data and handles imbalance nicely.

After training, I created a FastAPI service that can take transaction details and return a fraud probability. I packaged everything into a Docker container so the API runs the same way everywhere. The Dockerfile sits in the project root and includes all the files the app needs. Once the container is running, the API becomes available locally and can also be deployed to Azure Container Apps.

The project includes documentation, a user guide, and a final report explaining the whole process in a simple way. Overall, this was a great hands‑on experience building a real ML system that works end‑to‑end.
