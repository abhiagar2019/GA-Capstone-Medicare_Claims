<img src=images/Hospital-Ward-image-2.jpg width="875" height="500">

## Predicting patients with poor outcomes (hospital readmissions) from Medical Insurance Claims Data Only (No medical information except disease codes) - WIP

In this project, I am trying to predict patients with poor outcomes by using the frequency and duration of patient visit, their usage of prescription drugs and other products & services. I do not have access to any medical information such as vitals, test results, scans or any other kind of diagnostic tests. <i> Challenge is to predict worsening of patient's health without his/her detailed medical information.</i>


### Background
Overall 20% of the sickest patients consume 80% of the healthcare resources, be it cost or resource occupancy. Being able to predict the outcome (usually poor outcome) of a patient early-on can not only help in taking pre-emptive efforts to manage the condition but also help in managing the workload of the healthcare system thereby reducing cost and enhancing quality of life. 

### <i> Can the insurer with only limited medical information such as disease diagnosis code, billing for the equipments, services & drugs from medical claims be able to predict the poor outcome of its clients. I am attempting to predict this in this project.</i>

<img src=images/health_insurance.jpg width="265" height="200">

Steps:
1. Data Collection
2 a. Creating PostgreSQL database
2 b. Data Wrangling & Preparation
3. Feature Engineering
4. Exploratory Data Analysis  
5. Machine Learning Models (including dealing with class imbalance)
6. Setting up and running the models in Amazon Web Services (AWS) Tensorflow environment for Gridsearch
7. Hyper-Parameter Tuning 
8. Comparing all the classification model's performance 
9. Conclusion and Key learning
10. Future work


### 1. Data Collection
Data was collected from Center for Medicare and Medicaid Services (CMS), USA. It contains data for around 7m patients, with 1.3m inpatient, 16m outpatient claims and 111m prescription drug events and other carrier claims data for 2008, 2009 and 2010. 

<img src=images/CMS.png width="220" height="100"> 

I started with around a quarter of a million patients for exploratory data analysis. After seeing some trends between predictor and the target variables, finally I ended up using one-fifth of the available data. Prime reason for not using  the entire dataset was because of limitation of RAM on my machine and also to limit my expenditure on Amazon Web Services (AWS) while running the models.

 <img src=images/size.png width="300" height="150">
	

  ### 2 a. Creating PostgreSQL database
  
Dealing with such large data, especailly which required some cleaning, feature engineering and SQL joins (over 60 joins for the code lookups), it was best to create a SQL database.

  <img src=images/postgre_database.png width="500" height="300">
  
  ### 2 b. Data Wrangling & Preparation

As we know each patient can have multiple inpatient and outpatient visits. 

When a patient visits the medical facility, after initial assessment he/she is assigned a principal (or admission) diagnosis code which resulted in patient's admission. He/she is assigned other secondary diagnosis codes which include comorbidities, complications, and other diagnoses that are documented by the attending physician.

Usually each visit may result between 1 and 10 diagnose code, between 1 and 6 procedure codes, and between 1 and 45 HCPCS code.

There are around 20,000 (ICD9) disease, 13,000 HCPCS and 4,000 procedure codes to choose from.

<img src=images/pt1.png width="900" height="300"> 

Multiple visits of a single patient.
(Note: The raw data had diagnosis codes which I have joined to other lookup tables to get the description of each code)

So, when we join patient, inpatient claims, outpatient claims table using patient_id, we get multiple rows per patient, each row containing between 10 and 60 different diagnosis/HCPCS/Procedure codes. 

It is good to stitch the longitudinal view of the patient visits over a time-period and see how the disease progressed and/or new disease developed. 

One approach to view the diagnsos codes was to vectorize them and use each code as a separate feature. Using this approach, I got over 14,000 features (predictor variable) for my dataset. Postgre could handle at the max 1,600 features. So, I used <b> Principal Component Analysis (PCA) </b> to create a new feature space (eigen vectors) and identify the important features. Using PCA, I was able to reduce my features to approximately 600 features with over 82% cumulative variance.
  
  ### 3. Feature Engineering
  
  After inital explonatory data analysis, I calculated some fields such as:
  
  1. Length of stay (LOS)
  2. Readmissions within 30, 60 and 90 days
  3. Total number of inpatient admission
  4. Total number of outpatient visits
  5. Total number of diagnosis dosage 
  6. Total number of prescription dosage 
  7. Total medical equipment/supplies/services billed for
  8. Total cost incurred for a patient (insurer + copay + aid)
  9. Change in cost, number of visits, number of diagnosis from one year to another etc..
  
  <img src=images/summarized_by_patient.png width="900" height="250">
  
  ### 4.  Exploratory Data Analysis 
  
<b> Target variable is: Readmission within 30 days </b> 
  
  When plotted target variable against potential engineered features (Total number of inpatient admission and 
  Total number of diagnosis dosage), we can see a clear pattern that separates patients who are admiited vs non-admitted within 30 days of their inpatient visit.
  
   <img src=images/num_inpt_admissions.png width="425" height="300"> <img src=images/los.png width="425" height="300">
   
   <img src=images/total_inpt_diagnosis.png width="425" height="300"> <img src=images/total_inpt_procedures.png width="425" height="300">
   
   The correlation matrix and feature distributions showed meaningful relationship with the predictor variables. They also helped in identifying and getting rid of any co-linear features. 
  
   <img src=images/corelation_matrix.png width="900" height="500">
   
   <img src=images/feature_distribution.png width="900" height="500">
   
   All the features were standardized before running the models.
   
  ### 5. Machine Learning Models 

<b>  Baseline accuracy: 0.5 </b> 
 
<b> SMOTE over-sampling </b>  was used to deal with class imbalance (since there was class imbalance between patients who were re-admiited vs those who were not re-admitted within 30 days).
 
 All the following classification models were implemented.
<b>  
1. Logistic Regression
2. Polynomial Logistic Regression
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. KNN
7. Naïve Bayes
8. Support Vector Machine
9. Neural Network
</b>  

 ### 6. Setting up and running the models in Amazon Web Services (AWS) Tensorflow environment for Gridsearch
  
  All the base models with default settings were run on my local machine (macbook with 8GB RAM, quad-core). It took several hours for each model to run which made me choose AWS to run the Gridsearches for all the models.
  
 I uploaded the input files (X_train, y_train, X_test, y_test) to AWS Tensorflow which had all the required packages for me (except seaborn to plot ROC and precison-recall curve). 
 
 I gridsearched and downloaded the saved models back to my local computer (using <b> joblib </b> and <b> pickle </b> libraries).
  
  
 ### 7. Hyper-Parameter Tuning
  
  Extensive number of fits were performed for each model to identify the best parameters. Cross-validation and appropriate regularization parameter were chosen to avoid overfitting.
  
  Precision-recall, ROC curves and classfication reports were analysed to identify the best model.
  
  <img src=images/precision_recall.png width="400" height="250">
  
  Please note high false postives were favourable as we would be better off predicting more patients who needs attention than miss patients who may be performing worse in future.
  
  ### 8. Comparing all the classification model's performance 
  
  Comparing the models based on accuracy score. 
  
   <img src=images/scores.png width="700" height="500"> 
   
   Gradient Boosting comes out to be the top performing model. Although Random Forest seems to be reasonable good with 4 times faster run-time than Gradient Boosting.
   
   <img src=images/feature_imp_random_forest.png width="700" height="500"> 
   
 The feature importance from Random Forest that played vital role in prediction makes sense, as we have seen earlier in EDA. 
   
More Length of Stay (LOS), more number of inpatient admissions, more number of diseases (or co-morbidity) diagnosed, more number of inpatient procedures etc means higher chances of re-admission - > which is what the model has predicted quite well. 
   
 ### 9. Conclusion and key learning
1. Model accuracy increased for Logistic Regression (from linear to Polynomial)
2. Oversampling improved the accuracy (Polynomial Logistic Regression from 0.75 to 0.86)
3. PCA reduced the features from 14,000+ to 600 with 82% cumulative variance, but looses interpretability
4. Ensemble methods gave the best results
5. KNN was the slowest in this case so choosing appropriate parameters for Gridsearch is important

<b> It seems like poor outcome of patients (Re-admission within 30 days) can be predicted from claims only data (i.e. without much medical information) with reasonable accuracy. </b>
  
 ### 10. Future work
  Since now I have data in the right format, I would like to do the following projects:
  
1. Predicting Cost
2. Predicting Chronic Disease for a patient
3. Outcome-based reimbursement modelling - Bundling procedures & services (based on predicting future medical conditions) for bundled payments and taking preemptive measures
4. Flag high cost hospitals and physicians with poor outcomes
5. Since, I have zipcodes of the health providers where services was delivered, I would like to combine with othe data such as education, income equality, public transport, demographics to understand if and why certain zip-code population has poor health conditions
