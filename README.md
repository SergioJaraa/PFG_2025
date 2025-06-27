# PFG_2025 – Final Degree Project

This repository contains the code and resources developed for the 2025 Final Degree Project, focused on federated learning and result visualization using Streamlit.

## 🧠 Description

The main goal of this project is to implement a federated learning system that enables collaborative training of machine learning models without sharing local client data. Additionally, an interactive interface built with Streamlit is provided to facilitate the visualization and analysis of the obtained results.

## 📁 Repository Structure

- `FederatedLearning/`: Implementation of the federated learning system.
- `FineTuning/`: Contains the repository for fine tuning.
- `Data/`: Datasets used in the project.
- `Streamlit/`: Web application built with Streamlit for result visualization.
- `anaconda_projects/db/`: Configuration and databases used in Anaconda environments.
- `.vs/`: Development environment configuration files.
- `.gitignore`: Files and folders excluded from version control.
- `docker-compose.yml`: Docker configuration for containerized project execution.
- `README.md`: This file.

## 🚀 Requirements

- Python 3.10-slim
- Anaconda or Miniconda (optional but recommended)
- Docker (optional, for container execution)

## ⚙️ Installation and Execution

1. Clone this repository:

   ```bash
   git clone https://github.com/SergioJaraa/PFG_2025.git
   cd PFG_2025

2. To create an environment:
    ```bash
    conda create -n pfg_2025_env python=3.8
    conda activate pfg_2025_env
3. Installing dependencies from the Streamlit folder IMPORTANT:
   ```bash
   pip install -r requirements.txt
  
## To train a Federated Learning CNN
While on FederatedLearning folder:
1. Use one terminal to act as the server and execute:
   ```bash
   python server.py
2. To join a client, a terminal has to be opened and execute (Theres 3 clients):
   ```bash
   python client.py CLIENT_ID=client1

## Running the container with the interface:
1. Run this command in the PFG_2025 folder
   ```bash
   docker-compose up --build
2. Then, get into the url and enjoy
   Accessible at http://localhost:8501 (after running Docker)

## Author
SergioJaraa - Sergio Jaramillo Monasterio

