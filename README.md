# SafeJobFinder - Trust in your job searching

SafeJobFinder is a groundbreaking application developed to address a growing problem in the job market: the presence of fraudulent job offers. According to a recent report by the International Labour Organization, it is estimated that approximately 6% of job listings on online platforms are fraudulent or deceptive. These offers not only represent an economic risk but can also have serious consequences for the personal safety of applicants. In this context, SafeJobFinder emerges as an indispensable solution, using an advanced artificial intelligence-based classification model to identify and filter potentially fraudulent job postings. The application uses an extensive and diversified dataset, allowing for precise and effective identification of suspicious listings, thus contributing to a safer and more reliable job searching environment.

The effectiveness of SafeJobFinder is highlighted in its impressive accuracy of 95% and a ROC AUC score of 0.88, demonstrating its superior ability to distinguish between legitimate and fraudulent offers. These metrics are the result of a rigorous training and validation process of the model, employing cutting-edge machine learning techniques. The model's high accuracy minimizes the risk of mistakenly discarding legitimate offers, while its high ROC AUC value indicates an excellent capacity to detect a wide range of fraudulent tactics. These features make SafeJobFinder an essential tool not only for job seekers but also for online employment platforms seeking to maintain the integrity and trust in their services. With SafeJobFinder, users can explore job opportunities with the confidence that they are protected against deceit and scams in the digital job market.

This project is part of the Capstone Project for [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)




## Run Locally

Pre-requisites

```bash
python
git
docker
pandas
numpy
scikit-learm
streamlit
```
Activate docker desktop

Clone the project

```bash
  git clone https://github.com/Haroldgio28/fakejob_detection.git
```

Go to the project directory

```bash
  cd my-project
```

Build the docker image

```bash
  docker build --no-cache  -t job . 
```

Run the application using Docker

```bash
  docker run -it --rm -p 9696:9696 job
```

Run streamlit app from other command line and open, a window will open on web browser:

```bash
  streamlit run app.py
```

For make a fake job offer detection, there are two ways:

1. Running scripts or notebook from Visual Studio Code:
- Open the [notebook](https://github.com/Haroldgio28/fakejob_detection/blob/main/predict-test.ipynb) on Visual Studio Code or other code editor, and change the parameters.
- In another command line, execute the [script](https://github.com/Haroldgio28/fakejob_detection/blob/main/predict-test.py), changing the parameters on the script.

2. Modify the [file](https://github.com/Haroldgio28/fakejob_detection/blob/main/data/test.csv) and upload to streamlit app or insert data in the indicated fields.  
Look at the animation for guide.
   ![](https://github.com/Haroldgio28/fakejob_detection/blob/main/Animation.gif)


## Appendix

### Above the data
This dataset contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs. The dataset can be used to create classification models which can learn the job descriptions which are fraudulent. It has been sourced from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Haroldgio28)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/haroldgiovannyuribe/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/HaroldGio28)

 
