## Background
Controlling the coronavirus is undoubtedly one of the biggest global challenges scientists are immediately facing. There are currently over seven million confirmed cases in the United States, but the number of actual cases could be much higher due to lack of testing and false diagnoses. Some of the primary symptoms of Covid-19 include coughing and shortness of breath, which are both issues with the lungs. Chest CT scans have been successful in helping diagnose those infected, but it may take 1-2 days for patients to receive their results. With the number of cases still rising, it is crucial that we find a way to conduct a quicker, more accurate form of testing for the virus. More exact numbers would be extremely helpful in assisting us to understand the true severity of the situation, and how to combat the spread.

### Idea

The goal of this project is to help increase the speed that CT scans are analyzed while maintaining the diagnosis accuracy. Normally, a radiologist will compare two CT scans and make a diagnosis, but this process is one that scientists believe can be automated for better speed and accuracy. This can be done by processing images of lungs in healthy people vs those with the infection. A neural network needs to be constructed and will contain a trained model based on various lung images. Doctors will then be able to upload CT scans of their patient into this program and be given a diagnosis based on the predicted results.

### Current State

Incorporating AI into radiology is an area with a huge potential for growth. There are already numerous examples of image processing to conduct analysis on CT scans dealing with broken bones, respiratory illnesses, and other medical emergencies. As of October 2019, the FDA has approved 28 algorithms for image processing, but most of them are unique to individual facilities and are trained with small datasets. The Radiological Society of North America has created their own detection neural network for Covid-19 using lung images. Using data from six hospitals consisting of 4,536 x-ray images, they have designed a model that takes into account x-ray images, symptoms, patient age, and sex. **The design of this program will rely solely on image data and will expand the population of training images while being portable enough so that it can be widely utilized among healthcare professionals.**

### Development Steps and Timeline

 1. ***Data Collection and Cleaning: October 1st - October 17th***
    To create this, a convolutional neural network will need to be built using Keras and Tensorflow to analyze and classify the patient's CT scan. The model will be trained with a dataset containing images of lungs that are healthy, lungs from Covid-19 positive subjects, and lungs from other respiratory illnesses. Lung images of other respiratory illnesses need to be added in order to create a more accurate model. Without them, people who have miscellaneous lung problems would be falsely diagnosed with Coronavirus. Conveniently, researchers at the National Institute of Health have created a publicly accessible chest CT scan/x-ray database that contains over 100,000 images from more than 30,000 patients. The images have 14 labels associated with them, such as “No Finding”, “Pneumonia”, “Cardiomegaly”, “Edema”, etc. In addition to these illnesses, we need to add a label and image set for Covid-19, which will give us 15 labels. For this set we will be using the train images from confirmed Coronavirus CT scans. In order to confirm that the data collection was successful, we should make sure we can visualize each image and the corresponding label.

2.  ***Building the model: October 18th-November 1st***
    After we have collected the data, we can build the CNN model. It will be a sequential model optimized for this dataset using a Convolutional 2D layer, a Flatten layer, and a Dense layer with 15 total nodes. The dense layer will have softmax activation since this is a multi-classification model that will output a probability distribution. After compilation, we can train the model with the images and labels that we collected earlier. For the purpose of this program, the test image will be a single image from the second database, which has testing data for both positive and negative Covid-19 lungs. Using a single image from this set simulates how a doctor would use this tool to examine a lung CT scan for one patient.
3. ***Analyze and Display Results: November 1st-November 18th***
    The last step is evaluating our model to find the probability of each lung condition and predict the label in order to make a diagnosis based on the highest probability. To visualize the results, the test image should be printed next to the closest training image for comparison, along with what percent of a match it was (ex: 94.6% match).

## Technical Approach

 - Data Collection
	 - First, we gathered chest CT scans from open source databases.
	 - To receive an accurate prediction, we need a variety of respiratory illnesses to use as training data
		 - COVID-19
		 - Pneumonia
		 - Healthy
		 - Adenocarcinoma
		 - Large Cell Carcinoma
 - Preprocessing
	 - Images are preprocessed to make them the same shape.
	 - Next, they are grouped into training/test images.
	 - A label array gives each image a corresponding label to identify which illness group it belongs to.
	 ![Preprocessing phase](https://i.imgur.com/4h25i3J.png)

 - Building and Training the model
	 - Creating and building the model produces an untrained neural network (CNN in our case).
	 - Training the model uses the images and labels to "train" the data.
![Build and Train model](https://i.imgur.com/7Q3lWM9.png)
![basic neural network architecture](https://i.imgur.com/60TC6W4.png)
  
*Example of a basic neural network architecture*
 - Predict Results
	 - Using the trained model, we estimate the three most likely diagnoses along with their respective probabilities (e.g. COVID-19: 67%, Pneumonia: 14%, Adenocarcinoma: 9%, etc).
	
![Predict Results](https://i.imgur.com/bmCkDdO.png)

## Results

### Problems Encountered
Near the end of our project, we started to think about what kind of difficulties we ran into. The biggest roadblock we ran into was finding usable CT scans. We needed images that were certain formats (jpg, png, etc) but most of the datasets we discovered were of the DICOM standard. This is understandable as it's the standard format for medical imaging, but after further exploration we found that working with this format wasn't acceptable for our implementation (a single CT scan in DICOM format can be comprised of hundreds of various images).

Another interesting discovery we made is that many CT scans look similar to each other even though they present different respiratory illnesses. For example, a patient with COVID might have a CT scan that looks similar to a healthy patient and vice versa. In regard to training our model, this affected our results as each CT scan is patient dependent and lack critical context that would otherwise be needed to make a diagnosis (e.g. many patients develop pneumonia from COVID and it's near impossible to incorporate this contextual info into our model). Given our datasets for some respiratory illnesses were very small, we strongly believe our model would perform better given a much larger dataset.

## Project Materials

 - [Presentation Slides](https://docs.google.com/presentation/d/1nr-n63nrJ9jsreuh-RTk1XlCcX8rok-jeBqi4GhL3zY/edit?usp=sharing)
 - [Presentation Video](https://drive.google.com/file/d/1iFuPBp_VFopdzyHgY0gxxrm1S4wjgs4a/view?usp=sharing)
 - [Source Code on Github](https://github.com/sgronseth/CovidLungAnalysis)

## Tech Stack
 - Python
 - TensorFlow

### Sources
Below is a list of names and links to open source databases of CT scans used in this project.

[Zhang, Kang et al. “Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements, and Prognosis of COVID-19 Pneumonia Using Computed Tomography.” _Cell_ vol. 181,6 (2020): 1423-1433.e11. doi:10.1016/j.cell.2020.04.045](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7196900/)

[Adenocarcinoma and Large Cell Carcinoma CT Scans](https://www.kaggle.com/mohamedhanyyy/chest-ctscan-images/metadata)

[Zhao, Jinyu, et al. "COVID-CT-Dataset: a CT scan dataset about COVID-19."  arXiv preprint arXiv:2003.13865_  (2020).](https://arxiv.org/abs/2003.13865)

### Related Topics
Here are some projects that explore medical imaging's impact on COVID-19 research.

[Identifying COVID19 from Chest CT Images: A Deep Convolutional Neural Networks Based Approach](https://www.hindawi.com/journals/jhe/2020/8843664/)

[Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans](https://www.medrxiv.org/content/10.1101/2020.04.13.20063941v1)

[Artificial intelligence for the detection of COVID-19 pneumonia on chest CT using multinational datasets](https://www.nature.com/articles/s41467-020-17971-2)
