if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repors = "http://cran.us.r-project.org" )
if(!require(bayesplot)) install.packages("bayesplot", repors = "http://cran.us.r-project.org" )
if(!require(boot)) install.packages("boot", repors = "http://cran.us.r-project.org" )
if(!require(cowplot)) install.packages("cowplot", repors = "http://cran.us.r-project.org" )
if(!require(rJava)) install.packages("rJava", repors = "http://cran.us.r-project.org" )
if(!require(bartMachine)) install.packages("bartMachine", repors = "http://cran.us.r-project.org" )

library(dplyr)
library(tidyr)
library(tidyverse)
library(caret)
library(data.table)
library(randomForest)
library(bayesplot)
library(boot)
library(cowplot)
library(rJava)
library(bartMachine)
options(java.parameters = "-Xmx4g")
#You may need to increase the gig size of this java parameter so you don't run heap memory issues. 
#The plots will be easier to follow if you make your plot window much bigger now.  There are a lot of variables.  Unfortunately, they were produced very poorly in Rmarkdown. 

#The data is already split into two sets, a train set and a test set.  It consists of 81 variables and a total of 1456 objects composed of either character, categorical, numeric or integer type information.
#It is referred to as the Ames dataset. To contextualize, it is mostly composed of single family suburban dwellings that were sold in Ames, Iowa in the period 2006-2010.
#The data was obtained from kaggle.com. The links are below. 
#

#https://www.kaggle.com/datasets/rsizem2/house-prices-ames-cleaned-dataset?resource=download&select=clean_train.csv
#https://www.kaggle.com/datasets/rsizem2/house-prices-ames-cleaned-dataset?resource=download&select=clean_test.csv

#The goal is to use the train data to build a model to predict the SalePrice (the final variable) in the test set. 
#The purpose of this is to attempt to minimize both the RMSE and run-time in order to have a tool both accurate and efficient at assessing house costs.

#Make sure everything is being saved and retrieved from the correct location.
setwd("C:/Users/JoelS/Documents/projects/Capstone/Housecosts")
getwd()

#Ready the data for exploration 
dl <- tempfile()
download.file("https://raw.githubusercontent.com/storyhub/HousingCapstone/main/clean_train.csv", dl)
house_train <- read.csv(dl)
str(house_train)
dl2 <- tempfile()
download.file("https://raw.githubusercontent.com/storyhub/HousingCapstone/main/clean_test.csv", dl2)
house_test <- read.csv(dl2)
str(house_test)

head(house_train)
names(house_train)
dim(house_train)
summary(house_train) #many variables have a median of 0, but a non-zero mean and a relatively higher max.  This suggests there may few, but perhaps still relevant, occurances.

#Look at distribution of item of interest 
median(house_train$SalePrice) #The median house cost is 163000
mad(house_train$SalePrice) #The median absolute deviation is about 56000

#Let's look at this with a picture. 
house_train %>% 
  ggplot(aes(SalePrice/1000)) + geom_boxplot()

#Check for columns with NAs 
colSums(is.na(house_train))

#In the readme this value is described as "Linear feet of street connected to property". 
#Let's find out more about LotFrontage before we decide on what to do with the NAs
class(house_train$LotFrontage)
filter(house_train, house_train$LotFrontage == 0)
table(house_train$LotFrontage)


#It looks like there are no instances where LotFrontage = 0, so the NAs are likely instances where the property does not directly abut a street.
#We will change those instances to a quantity, 0, rather than leave it as NA. 
house_train <- house_train %>% mutate_at(vars("LotFrontage"), ~replace_na(.,0))
is.na(house_train$LotFrontage)
#Success, the NAs have been replaced by 0's.

#plot by category
distribution_plots <- lapply(names(house_train), function(x){
  plot <- 
    ggplot(house_train) +
    aes_string(x)
  
  if(is.numeric(house_train[[x]])) {
    plot <- plot + geom_histogram()
    
  } else {
    plot <- plot + geom_bar()
  } 
  
})

plot_grid(plotlist = distribution_plots)


#plot character variables
character_var <- house_train %>% dplyr::select_if(is.character)
  cv_list <- names(character_var)

  distribution_character <- lapply(cv_list, function(x){
    plot <- 
      ggplot(house_train) +
      aes_string(x)
  plot <- plot + geom_bar() + theme(axis.text.x = element_text(angle = 90))
    
  })
  
  plot_grid(plotlist = distribution_character)
  
  
  #plot integer variables
integer_var <- house_train %>% dplyr::select_if(is.integer)
  int_list <- names(integer_var)  
  
  distribution_integer <- lapply(int_list, function(x){
    plot <- 
      ggplot(house_train) +
      aes_string(x)
    plot <- plot + geom_histogram() + theme(axis.text.x = element_text(angle = 90))
    
  })
  
  plot_grid(plotlist = distribution_integer)
  
  #plot numeric variables
numeric_var <- house_train %>% dplyr::select_if(is.numeric)
  num_list <- names(numeric_var)

  distribution_numeric <- lapply(num_list, function(x){
    plot <- 
      ggplot(house_train) +
      aes_string(x)
    plot <- plot + geom_histogram() + theme(axis.text.x = element_text(angle = 90))
    
  })
  
  plot_grid(plotlist = distribution_numeric)

  #There are many variables that appear to have very low variance. 

nnz_var <- nearZeroVar(house_train, names = TRUE)
nnz_var
nnz_noname <- nearZeroVar(house_train)
nnz_noname

  #Here's a visual
distribution_nnz <- lapply(nnz_var, function(x){
  plot <- 
    ggplot(house_train) +
    aes_string(x)
  plot <- plot + geom_dotplot() + theme(axis.text.x = element_text(angle = 90))
  
})

plot_grid(plotlist = distribution_nnz)
  

#"quality" assessment variables
quality_vars <- house_train %>% ls(pattern = "Q")
cat(quality_vars)
condition_vars <- house_train %>% ls(pattern = "Cond")
#Condition 1 and 2 and Sale condition are not quality assessments. We'll remove those from the list. 
cat(condition_vars)
str(condition_vars)
condition_vars <- condition_vars[condition_vars != "Condition1"]
condition_vars <- condition_vars[condition_vars != "Condition2"]
condition_vars <- condition_vars[condition_vars != "SaleCondition"]

subjective_vars <- append(quality_vars, condition_vars)
cat(subjective_vars)

#Sometimes character data can be problematic, we'll change it to factors.
house_train <- house_train %>% mutate_if(is.character, as.factor)

#Split training set to train and test sets
house_index <- createDataPartition(y = house_train$SalePrice, times = 1, p = 0.2, list = FALSE)
house_trainer <- house_train[-house_index,]
small_test <- house_train[house_index,]
dim(house_trainer)
dim(small_test)
str(house_trainer)
str(small_test)

#The 81st column is the SaleSprice 
head(house_train[81])

#establish definition of RMSE
RMSE <- function(true_price, predicted_price){
  sqrt(mean((true_price - predicted_price)^2))
}
true_price <- small_test$SalePrice

#Develop a random forest model.
#Parameters to help control for low variance variables and highly correlated variables have been set to help reduce computational resources for parameters with little value. For now the ntree value is kept low to reduce computational expense. 
random_house <- train(house_trainer[-81], house_trainer$SalePrice, method = "rf", ntree = 50, preProcess = c("zv", "corr"), trControl = trainControl(method = "cv", number = 25, p = .9))

#check results
min(random_house$results$RMSE)
random_house$finalModel


#check which variables appear most important. 
varImp(random_house)
#This took about 7 min to run.  Let's see if we can reduce the number of variables to improve runtime without losing much predictive value. 
plot(varImp(random_house))
#The following were the top variables of importance, in the double digits in importance score.
# OverallQual  100.000
#GrLivArea     39.625
#ExterQual     35.910
#GarageCars    34.285
#TotalBsmtSF   20.368
#X1stFlrSF     16.496
#BsmtQual      16.013
#KitchenQual   14.790
#YearBuilt     13.440

importantvariables <- house_trainer %>% dplyr::select(OverallQual, GrLivArea, ExterQual, GarageCars, TotalBsmtSF, X1stFlrSF, BsmtQual, KitchenQual, YearBuilt)
print(importantvariables)
#The model will be rewritten using the above variables only.
random_house2 <- train(importantvariables, house_trainer$SalePrice, metric = "RMSE", method = "rf", ntree = 50, preProcess = c("corr"), trControl = trainControl(method = "cv", number = 25, p = .9))
#runtime, ~2 min.
min(random_house2$results$RMSE)
varImp(random_house2)$importance

random_pred2 <- predict(random_house2, small_test)
RMSE(true_price, random_pred2)
#It looks like we are having on average a about a $2000-$4000 dollar loss in price prediction with the reduced variable model.  Lets compare time to predictive value by varying the number of trees. 
#Model with many variables, but small ntree.  It took about 4 min to run. 
treestand_house <- train(house_trainer[-81], house_trainer$SalePrice, method = "rf", ntree = 25, preProcess = c("zv", "corr"), tuneLength = 25, trControl = trainControl(method = "cv"))
min(treestand_house$results$RMSE)

#Model with few variables, but large ntree.  It took about 1.5 min to run. 
vegetative_house <-  train(importantvariables, house_trainer$SalePrice, metric = "RMSE", method = "rf", ntree = 200, preProcess = c("corr"), tuneLength = 25, trControl = trainControl(method = "cv"))
min(vegetative_house$results$RMSE)

#add back in more variables.  Maybe reduce number of trees? 
more_variables <- house_trainer %>% dplyr::select(OverallQual, GrLivArea, ExterQual,
                                           GarageCars,
                                           X1stFlrSF,
                                           GarageArea,
                                           TotalBsmtSF,
                                           BsmtQual,
                                           KitchenQual,
                                           BsmtFinSF1,
                                           LotArea,
                                           TotRmsAbvGrd,
                                           YearBuilt,
                                           X2ndFlrSF,
                                           FireplaceQu,
                                           GarageFinish,
                                           FullBath,
                                           MasVnrArea,
                                           YearRemodAdd)
random_house3 <-  train(more_variables, house_trainer$SalePrice, metric = "RMSE", method = "rf", ntree = 200, preProcess = c("corr"), tuneLength = 25, trControl = trainControl(method = "cv"))
min(random_house3$results$RMSE)
varImp(random_house3)$importance

#This model performed similar to the first model, and was a little faster, running in about 5 minutes.  We'll try one more to time for a more balanced result. 
final_variables <- house_trainer %>% dplyr::select(OverallQual, GrLivArea, ExterQual,
                                                  GarageCars,
                                                  X1stFlrSF,
                                                  GarageArea,
                                                  TotalBsmtSF,
                                                  BsmtQual,
                                                  KitchenQual,
                                                  BsmtFinSF1,
                                                  LotArea,
                                          -        TotRmsAbvGrd,
                                                  YearBuilt,
                                                  X2ndFlrSF,
                                                  FireplaceQu,
                                                  FullBath,
                                                  YearRemodAdd)


final_random_house <-  train(final_variables, house_trainer$SalePrice, metric = "RMSE", method = "rf", ntree = 150, preProcess = c("corr"), tuneLength = 25, trControl = trainControl(method = "cv"))
min(final_random_house$results$RMSE)
varImp(final_random_house)$importance
plot(final_random_house)
final_random_house$finalModel

#This model ran in about 3 minutes had similar predictive value to our initial model with an RMSE of around 20000.  This is maybe not so useful to individuals on a budget looking for a bargain, but perhaps could serve those looking at higher cost purchases. 

#Try the BartMachine instead
#first two 45 sec
machine_house <- bartMachine(house_trainer[-81], house_trainer$SalePrice, num_trees = 75, impute_missingness_with_rf_impute =TRUE, mem_cache_for_speed = FALSE, flush_indices_to_save_RAM = TRUE)
summary(machine_house)
machine_house_more <- bartMachine(house_trainer[-81],house_trainer$SalePrice, num_trees = 100, impute_missingness_with_rf_impute =TRUE, mem_cache_for_speed = FALSE, flush_indices_to_save_RAM = TRUE)
summary(machine_house_more)
#35 sec
machine_house_less <- bartMachine(house_trainer[-81], house_trainer$SalePrice, num_trees = 60, impute_missingness_with_rf_impute =TRUE, mem_cache_for_speed = FALSE, flush_indices_to_save_RAM = TRUE)
summary(machine_house_less)
#60 sec
machine_house_many <- bartMachine(house_trainer[-81], house_trainer$SalePrice, num_trees = 200, impute_missingness_with_rf_impute =TRUE, mem_cache_for_speed = FALSE, flush_indices_to_save_RAM = TRUE)
summary(machine_house_many)

bart_choices <- investigate_var_importance(machine_house_more, num_replicates_for_avg = 20)
head(bart_choices$avg_var_props, n = 15L)

#Quality only- If you were to attempt to generalize this for personal use, it may be difficult to obtain much of this data.  Perhaps if you could score quality on a number of aspects you could use this to help decide a good price.
quality_data <- house_trainer %>% dplyr::select(subjective_vars)
str(quality_data)
subjective_bart <- bartMachine(quality_data, house_trainer$SalePrice, num_trees = 100, impute_missingness_with_rf_impute =TRUE, mem_cache_for_speed = FALSE, flush_indices_to_save_RAM = TRUE)
summary(subjective_bart)
#this is much worse


#remove low variance variables 
bart_variables <- house_trainer %>% dplyr::select(-c(nearZeroVar(house_trainer)))
dim(bart_variables)
rv_bart_house <- bartMachine(bart_variables[-62], house_trainer$SalePrice, num_trees = 100, impute_missingness_with_rf_impute =TRUE, mem_cache_for_speed = FALSE, flush_indices_to_save_RAM = TRUE)
summary(rv_bart_house)
#It was about 10 seconds faster, but about performed somewhat worse. We'll CV with the origianl model as there is less likelihood of overfitting.


#cross-validate, ~2.5 min
bart_cv <- function(bart_variables, ind){
  data <- bart_variables[ind,] 
  output <- bartMachine(house_trainer[-81], house_trainer$SalePrice, num_trees = 100, impute_missingness_with_rf_impute =TRUE, mem_cache_for_speed = FALSE, flush_indices_to_save_RAM = TRUE)
  sqrt(mean((resid(output))^2)) 
}
bcv_output <- boot(data = bart_variables, statistic = bart_cv, R = 10)
plot(bcv_output)
#the results appear similar to the original model.

#impute_missingness_with_rf_impute =TRUE was attempted to be used to assist with the slight differences in low frequency occurrences, but predict complained of row differnces, so it had to be removed. 
final_bart_house <- bartMachine(house_trainer[-81], house_trainer$SalePrice, num_trees = 100, mem_cache_for_speed = FALSE, flush_indices_to_save_RAM = TRUE)


#test the models against the holdout data
#RF
final_random_pred <- predict(final_random_house, small_test)
RMSE(true_price, final_random_pred)

#BART
bart_pred <- bart_predict_for_test_data(final_bart_house, small_test[-81], small_test$SalePrice)
bart_pred$rmse


#Conclusion
#See RMD-PDF for larger summary. 
#RMD file did not knit quite like I was expecting, producing a lot of un-useful text for a summary.  
#I will need to learn how to use Rmarkdown better to get the plots and text produced in an easier to follow fashion. 