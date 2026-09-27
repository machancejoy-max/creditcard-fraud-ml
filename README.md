Credit Card Fraud Detection – End‑to‑End ML System
This project implements a complete machine learning pipeline for detecting credit card fraud using XGBoost, FastAPI, Docker, and Azure Container Apps. It includes data exploration, preprocessing, model training, deployment, and automated CI/CD workflows.

Project Overview
The goal of this project is to build a scalable, production‑ready fraud detection system. The pipeline covers:

Exploratory Data Analysis

Data preprocessing and class imbalance handling

Model development and tuning

API creation for real‑time predictions

Containerization and cloud deployment

Automated CI/CD for continuous updates

Strategy for model retraining and maintenance

Key Features
End‑to‑end ML workflow

XGBoost classifier optimized for imbalanced data

SMOTE oversampling

FastAPI prediction service

Dockerized application

Deployment on Azure Container Apps

GitHub Actions CI/CD pipeline

Saved model and scaler for consistent inference

Repository Structure
notebooks/ – Exploratory data analysis

src/ – ML pipeline, API, utilities, and saved models

data/ – Dataset (if included)

Dockerfile – Container definition

requirements.txt – Dependencies

Deployment
The application is deployed using Azure Container Apps and exposes a public endpoint for fraud prediction. The Docker image is stored in Azure Container Registry, and GitHub Actions automates testing, building, and deployment.

Model Maintenance
A retraining strategy ensures long‑term model performance. The system supports periodic retraining and redeployment when new data becomes available or when performance metrics decline.

Status
The project is fully implemented, deployed, and version‑controlled. It includes documentation, a user guide, and a final report summarizing the entire workflow.
