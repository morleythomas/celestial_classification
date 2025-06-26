Readme.txt

This project is to classify objects from astronomical surveys as galaxy/uasar/stellar object and so on. Quite simple, but should be a quick one and a good start and can upload to GitHub. Finish in like a couple days ?

Link to Kaggle: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17?resource=download

Check Kaggle link and just see what others are doing with it! But also, play around with a bit first! Data augmentation, different models, random forests and what not! 

Idea: 
- notebook 1: test different model (NN, RF, something else more exotic...?)
- notebook 2: data augmentation? is that possible? Do expiatory data analysis - for instance, is it possible that we should be doing PCA on multicolinear variables

Produce a poetry project with source code that contains all the code, so that the notebook can literally be as basic and easy to read as possible !
Use poetry to produce a package that we can import for experiments


Tasks:
- explore data ✅
- first round of experiments will involve default methods for classifiers (these will prove an initial baseline, ie how far can we get without tuning anything?)
- explore random forrest
- explore decision tree
- explore neural network
- explore naive bayes
- explore xgboost
- explore k nearest neighbours
- explore logistic regression
- IMPORTANT: wtf is naive bayes agin??
- then, we select the three best ones and adjust hyper parameters for fine tuning
- then, we can use data augmentation (by sampling from a gaussian distribution defined by each feature's mean and td. deviation) to see if there is a statistically significant improvement in results (think about, how will we define accuracy (f1 score and ROC curve?) and how can we measure uncertainty in the f1 score, to have confidence in our claim of improving over historic baseline (uncertainty quantification/power analysis))



Problem statement.
We want to know if we can improve upon using default algorithms to make predictions on the stellar dataset using data augmentation. This means, we produce initial baselines based on the performance of default methods.
We then select the best methods and see if their results improve post augmentation processing. We define success as a statistically significant improvement of accuracy when using augmented data over the 
initial baseline.


Dummy baseline (Accuracy):
57% using weighted random prediction (.13% std. error)
59% using most frequent class prediction

Initial baseline (Accuracy):



Data
obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS

####### These two are coordinates - make sense to combine them? #########
alpha = Right Ascension angle (at J2000 epoch)
delta = Declination angle (at J2000 epoch)

####### Wavelength filters - combine? #########
u = Ultraviolet filter in the photometric system
g = Green filter in the photometric system
r = Red filter in the photometric system
i = Near Infrared filter in the photometric system
z = Infrared filter in the photometric system

####### Not sure if these will be informative for classification? ######### if it turns out these are correlated with the answers, is there maybe some issue with the instrument. ie, it should not really be relevant info
run_ID = Run Number used to identify the specific scan
rereun_ID = Rerun Number to specify how the image was processed
cam_col = Camera column to identify the scanline within the run
field_ID = Field number to identify each field

######### Not sure how to use this ? #########
spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)

class = object class (galaxy, star or quasar object)
redshift = redshift value based on the increase in wavelength

######### not sure if useful? ######### if it turns out these are correlated with the answers, is there maybe some issue with the instrument. ie, it should not really be relevant info
plate = plate ID, identifies each plate in SDSS
MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation


The data we want to use:
- redshift
- u
- g
- r
- i
- z
- spec_obj_ID // suspicious as to whether this belongs with the two below 
- plate // could be circumstantial though, and inclusion might overfit to the particular dataset
- MJD // could be circumstantial though, and inclusion might overfit to the particular dataset



