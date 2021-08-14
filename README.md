# Ames Housing Data Project
![alt text](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/logos/header.png?t=2021-04-23-18-16-53)

# Problem Statement
Based on the Ames Housing Dataset, create a regression model that will predict the price of a house at sale.


## Content
- [Data Import & Cleaning](#Data-Import-and-Cleaning)
    * [Missing Value](#Missing-Value)
    * [Imputing Ordinal Variable](#Imputing-Ordinal-Variable)
    * [Imputing Nominal Variable](#Imputing-Nominal-Variable)  
- [Feature Engineering/Selection](#Feature-Engineering/Selection)
    * [Distribution of target variable](#Distribution-of-target-variable)
    * [Correlation with Target Predictor](#Correlation-with-Target-Predictor)
    * [Transforming Age](#Correlation-with-Target-Predictor)
    * [Feature Selection of Correlation with Target Predictor](#Feature-Selection-of-Correlation-with-Target-Predictor)    
- [Model Fitting](#Model-Fitting)
    * [Lasso](#Lasso)
    * [with PolyNominal Feature](#with-PolyNominal-Feature)
- [Model Inference](#Model-Inference)
    * [Predicted vs True Value](#Predicted-vs-True-Value)
    * [Model](#Model)
    * [Conclusion](#Conclusion)

## Data Dictionary

This data is provided by General Assembly.

|Feature|Type|Description|
|:---:|:---:|:---|
|**ID**|*Discrete*| Observation number|
|**PID**|*Nominal*|Parcel identification number  - can be used with city web site for parcel review|
|**MS SubClass**|*Nominal*|Identifies the type of dwelling involved in the sale|
|**MS Zoning**|*Nominal*|Identifies the general zoning classification of the sale|
|**Lot Frontage**|*Continuous*|Linear feet of street connected to property|
|**Lot Area**|*Continuous*|Lot size in square feet|
|**Street**|*Nominal*|Type of road access to property|
|**Alley**|*Nominal*|Type of alley access to property|
|**Lot Shape**|*Ordinal*| General shape of property|
|**Land Contour**|Nominal*|Flatness of the property|
|**Utilities**|*Nominal*|Type of utilities available|
|**Lot Config**|*Nominal*|Lot configuration|
|**Land Slope**|*Ordinal*|Slope of property|
|**Neighborhood**|*Nominal*|Physical locations within Ames city limits (map available)|
|**Condition 1**|*Nominal*|Proximity to various conditions|
|**Condition 2**|*Nominal*|Proximity to various conditions(if more than one is present)|
|**Bldg Type**|*Nominal*|Type of dwelling|
|**House Style**|*Nominal*|Style of dwelling|
|**Overall Qual**|*Ordinal*|Rates the overall material and finish of the house|
|**Overall Cond**|*Ordinal*|Rates the overall condition of the house|
|**Year Built**|*Discrete*|Original construction date|
|**Year Remod/Add**|*Discrete*|Remodel date (same as construction date if no remodeling or additions)|
|**Roof Style**|*Nominal*|Type of roof|
|**Roof Matl**|*Nominal*|Roof material|
|**Exterior 1**|*Nominal*|Exterior covering on house|
|**Exterior 2**|*Nominal*|Exterior covering on house(if more than one material)|
|**Mas Vnr Type**|*Nominal*|Masonry veneer type|
|**Mas Vnr Area**|*Continuous*|Masonry veneer area in square feet|
|**Exter Qual**|*Nominal*|Evaluates the quality of the material on the exterior|
|**Exter Cond**|*Ordinal*|Evaluates the present condition of the material on the exterior|
|**Foundation**|*Nominal*|Type of foundation|
|**Bsmt Qual**|*Ordinal*|Evaluates the height of the basement|
|**Bsmt Cond**|*Ordinal*|Evaluates the general condition of the basement|
|**Bsmt Exposure**|*Nominal*|Refers to walkout or garden level walls|
|**BsmtFin Type 1**|*Nominal*|Rating of basement finished area|
|**BsmtFin SF 1**|*Continuous*|Type 1 finished square feet|
|**BsmtFinType 2**|*Ordinal*|Rating of basement finished area (if multiple types)|
|**BsmtFin SF 2**|*Continuous*|Type 2 finished square feet|
|**Bsmt Unf SF**|*Nominal*|Unfinished square feet of basement area|
|**Total Bsmt SF**|*Continuous*|Total square feet of basement area|
|**Heating**|*Nominal*| Type of heating|
|**HeatingQC**|*Ordinal*|Heating quality and condition|
|**Central Air**|*Nominal*| Central air conditioning|
|**Electrical**|*Ordinal*|Electrical system|
|**1st Flr SF**|*Continuous*|First Floor square feet|
|**2nd Flr SF**|*Continuous*|Second floor square feet|
|**Low Qual Fin SF**|*Continuous*|Low quality finished square feet (all floors)|
|**Gr Liv Area**|*Continuous*|Above grade (ground) living area square feet|
|**Bsmt Full Bath**|*Ordinal*|Basement full bathrooms|
|**Bsmt Half Bath**|*Continuous*|Basement half bathrooms|
|**Full Bath**|*Discrete*|Full bathrooms above grade|
|**Half Bath**|*Discrete*|Half baths above grade|
|**Bedroom**|*Discrete*|Bedrooms above grade (does NOT include basement bedrooms)|
|**Litchen**|*Discrete*|Kitchens above grade|
|**KitchenQual**|*Ordinal*|Kitchen quality|
|**TotRmsAbvGrd**|*Discrete*|Total rooms above grade (does not include bathrooms)|
|**Functional**|*Ordinal*|Home functionality (Assume typical unless deductions are warranted)|
|**Fireplaces**|*Discrete*|Number of fireplaces)|
|**FireplaceQu**|*Ordinal*|Fireplace quality|
|**Garage**|*Nominal*|Garage location|
|**Garage Yr Blt**|*Discrete*|Year garage was built|
|**Garage Finish**|*Ordinal*|Interior finish of the garage|
|**Garage Cars**|*Discrete*|Size of garage in car capacity|
|**Garage Area**|*Continuous*|Size of garage in square feet|
|**Garage Qual**|*Ordinal*|Garage quality|
|**Garage Cond**|*Ordinal*| Garage condition|
|**Paved Drive**|*Ordinal*|Paved driveway|
|**Wood Deck SF**|*Continuous*|Wood deck area in square feet|
|**Open Porch SF**|*Continuous*|Open porch area in square feet|
|**Enclosed Porch**|*Continuous*|Enclosed porch area in square feet|
|**3-Ssn Porch**|*Continuous*|Three season porch area in square feet|
|**Screen Porch**|*Continuous*|Screen porch area in square feet|
|**Pool Area**|*Continuous*| Pool area in square feet|
|**Pool QC**|*Ordinal*|Pool quality|
|**Fence**|*Ordinal*| Fence quality|
|**Misc Feature**|*Nominal*|Miscellaneous feature not covered in other categories|
|**Misc Val**|*Continuous*|Value of miscellaneous feature|
|**Mo Sold**|*Discrete*|Month Sold (MM)|
|**Yr Sold**|*Discrete*| Year Sold (YYYY)|
|**Sale Type**|*Nominal*|Type of sale|
|**Sale Condition**|*Nominal*|Condition of sale|
|**SalePrice**|*Continuous*|Sale price|


# Data Import and Cleaning

## Missing Value

There is high missingness in _Pool QC, Misc Feature, Alley,Fence,Fireplace & Lot Frontage_ and some level of missingness across other feature.

In the DataDocument, the missingness is **expected** as it indicates the absence of the feature. For example, for a house that has no Garage, the _Garage Quality_ feature will be NA.

But how can one tell if NA is absence of the feature or a missing value?
<br>The missingness is consistence across the Main Category of a feature, eg for *Garage*, it can be seen that _Garage Qual, Garage Yr Bl, Garage Type_ missingness are consistence.
Hence these N/A shall NOT be treated as missing value.

**Action**:

1. For _Pool QC, Misc Feature, Alley_, Although it is not a missing value, but in terms of feature variance, it is extremelty low,Hence,  we can safety drop it as low variance will not be a good predictor for our target.  


2. For Discrete Value, we will fill the N/A as 0. However for certain Discrete Value, eg Garage Yr Blt, we shall be careful as filling 0 have different intepretation (eg the garage is built recently). We shall check *Garage Yr Blt* is having high correlation with bldg *Year Built*, we can drop the *Garage Yr Blt* Column.  


3. For Numerical(Continuous Category, We will fill the N/A with 0. For Ordinal & Nominal, we need to fill a string type as OneHotEncoder will not work with mixed data type column.

## Imputing Ordinal Variable

Normally we will perform OrdinalEncoder (and also OneHotEncoder) on Train set only to avoid data leakage. However, in this dataset it is assumed that all ordinal category variables are listed as per the documentation and therefore there will not be any future unseen value.

## Imputing Nominal Variable

Nominal Variable will be encoded via OneHotEncoding

# Feature Engineering/Selection

## Distribution of target variable

<a href="https://ibb.co/2vhWJV5"><img src="https://i.ibb.co/qNj76qx/Distribution-of-Target.jpg" alt="Distribution-of-Target" border="0"></a>

The target Variable is having positive skewness. We will need to transform it to normal distribution to be able to use Linear Regression.

## Distribution of Continuous Variable

<a href="https://ibb.co/92W8Tkp"><img src="https://i.ibb.co/ZKxL8pf/Distribution-of-Continuous-Variable.jpg" alt="Distribution-of-Continuous-Variable" border="0"></a>

We can see that **_Garage Area, Gr Liv Area, Total BSMF, 1st Flr SF_** is close to normal distribution, which will have higher correlation with target variable Sale Price and can be a good predictor. We shall investigate on correlation to confirm this. However intuition says that Mo Sold does not really matters with price. We will investigate below to further confirm

## Distribution of Discrete Variable

<a href="https://ibb.co/37VhKtT"><img src="https://i.ibb.co/L8BpK40/Distribution-of-Discrete-Variable.jpg" alt="Distribution-of-Discrete-Variable" border="0"></a>

We can see that **_Year Built, TotRms Abv Grd & Mo Sold_** have close to normal distribution which will have higher correlation with target variable Sale Price and can be a good predictor. We shall investigate on correlation to confirm this. However intuition says that Mo Sold does not really have a meaningful relationship with price.

## Distribution of Ordinal Variable

<a href="https://ibb.co/7g6rMyp"><img src="https://i.ibb.co/WDYB8Kz/Distribution-of-Ordinal-Variable.jpg" alt="Distribution-of-Ordinal-Variable" border="0"></a>

We can see that **_Overall Qual, Exter Cond_** have close to normal distribution. which will have higher correlation with target variable **Sale Price** and can be a good predictor. We shall investigate on correlation to confirm this. However intuition says that Mo Sold does not really have a meaningful relationship with price.

## Correlation with Target Predictor

<a href="https://ibb.co/J7WpqT4"><img src="https://i.ibb.co/r6Kp3hS/Correlation.jpg" alt="Correlation" border="0"></a>


It is seen that overall top 10 of the features is having positive correlation with the target, while some of the feature is having negative correlation. However, interesting the negative correlation is unexpected for **_Kitchen AbvGr, Bedroom AbvGr, Overall Cond_**, especially **_Overall Cond_**, where the higher it is, the pricier the housing should be. However note that the correlation is not that strong (approx negative 0.1) it may be due to randomness that lead to this slight negative correlation

## Transforming age

We have transform Yr Built and Yr Sold to Age of the house. However, the transformation did not have a better result. It is almost same as year-Built. For better inference model. we will keep that transform the age when sold feature.


# Model Fitting

## Lasso

For first model, We will fit a Lasso model with following steps

1. Age transformation
2. getting correlated feature with specific threshold
3. Scaling of feature
4. transforming target to have normal distribution
5. Fitting Lasso model
6. Hyperparameter tuning for best parameter


From results, it can be seen that when the correlation threshold is lower(more feature) the better the test score. However the different between training set and test set is wider. This is a sign of overfitting when there is more feature.
From the above result we will select threshold = 0.7 and alpha = 10  with absolute different of 1% between training and testing as our hyperparameter for model training.

### Model Evaluation with test set

With RMSE of test set of 41312, the difference with training set (39882) is appoximately 3.5%

## with PolyNominal Feature

We have fitted polynominal of degree two.

Training score returns 33224 RMSE but test score returns 46030 RMSE, there is a severe overfitting for model_2 (with polynomianl feature), even if we fine tune the alpha with Lasso, the model will not be any better compare to previous model and it adds complexity for inference. Hence, we will retain our model_1 as our final model.

# Model Inference

## Comparing prediction vs True Value

<a href="https://ibb.co/L0BCdyJ"><img src="https://i.ibb.co/9pBqH0V/Predicted-vs-True-Values.jpg" alt="Predicted-vs-True-Values" border="0"></a>

1. The model predicted accurately at the lower range of housing value, except for one extreme outlier.  The model prediction under predict at values higher than 300,000 of predicted value

2. On the "outlier", our model has over predicted the housing value. The "outlier" can be truly an outlier (eg: a mistake by data entry into the feature) , or it contains certain fea### RMSE without the Outlier

### RMSE without the Outlier

The RMSE without the outlier is approx 29158 as comapre to earlier 46030. Hence, in future production, we could inform stake holder that the model can predict at accuracy of $ 29158 if we validate our predicted value within certain range.

## Model

sqrt(Housing Sale Price) = 37.89 * Overall Qual + 13.11 * Exter Qual + 24.5 * Gr Liv Area

Based on F-statistical test, the above model is significant.

## Conclusion

We have created a regression model which takes inputs of _Overall Qual, Exter Qual & Gr Liv Area_ in  which the predicted accuracy is good at predicting lower house price range. However it accuracy is worsen at above  300,000 dollar.
<br> The model predicts a house price with accuracy of +- \\$ 29158 when the prediction outputs value ( < \\$ 300,000), however cautious must be given when the predicted value is above \\$ 300,000
