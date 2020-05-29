<img src=images/Hospital-Ward-image-2.jpg width="700" height="500">

## Predicting patients with poor outcomes (hospital readmissions) from Medical Insurance Claims Data 

In this project, I am trying to predict patients with poor outcomes by using the frequency and duration of patient visit, their uasge of prescription drugs and other products & services. I do not have access to any medical information such as vitals, test results, scans or any other kind of diagnostic tests.


### Background
Overall 20% of the sickest patients consume 80% of the healthcare resources, be it cost or resource occupancy. Being able to predict the outcome (usually poor outcome) of a patient early-on can not only help in taking pre-emptive efforts to manage the condition but also help in managing the workload of the healthcare system thereby reducing cost and enhancing quality of life. 

### Can the insurer with only limited medical information such as disease diagnosis code, billing for the equipments, services and drugs used be able to predict the poor outcome of its clients. This project is an attempt to try this.

<img src=images/health_insurance.jpg width="400" height="300">


Steps:
1. Data Collection
2. Data Wrangling & Preparation
  2 a. Creating PostgreSQL database
3. Exploratory Data Analysis 
4. Feature Engineering
5. Machine Learning Models 
  5 a. Setting up and running the models in Tensorflow environment in Amazon Web Services (AWS)
6. Hyper parameter tuning (including dealing with class imbalance)
7. Comparing all the classification model's performance 
8. Conclusion and future work


### 1. Data Collection
Data was collected from Center for Medicare and Medicaid Services (CMS), USA

<img src=images/CMS.png width="200" height="100">
	
### 2. Data Wrangling & Preparation

 <img src=images/size.jpg width="400" height="200">
 
1. Each patient can have multiple inpatient and outpatient visits
2. Each visit can have multiple ICD9 (diagnosis) assigned for each visit
3. There are 20,000 disease, 13,000 HCPCS and 4,000 procedure codes
 
  ### 2 a. Creating PostgreSQL database
  
  <img src=images/postgreSQL.png width="400" height="200">
  
#### poor outcomes of (without much medical information about the patients? 

Prediciting Poor Outcomes from Medical Insurance Claims data with very limited medical information. 
