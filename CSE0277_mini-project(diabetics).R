#---------------------------MINI PROJECT-------------------------------------------- 


#---------------PIMA INDIAN DIABETES PREDICTION DATASET--------------------------



#This data set is analysed in R using 04 algorithms for the prediction of diabetic in pregnant women:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. Support Vector Machine (SVM) and
# 5. Comparison of Model Accuracy.




#Naming the dataset as data
data <- diabetes2
View(data)

#Retrieves first 5 records
head(data)

#Retrieves last 5 records
tail(data)

#About DataFrame structure of the data
str(data)

#Dimensions
dim(data)

#checking if missing values
colSums(is.na(data))

#if it has null value 
data$Glucose[is.na(data$Glucose)] = median(data$Glucose, na.rm=TRUE)
colSums(is.na(data))

summary(data)


#we need to make target(outcome) as factor(class)
data$Outcome <- as.factor(data$Outcome)
summary(data)


d=levels(data$Outcome) <- c("Non-diabetic","Diabetic")
head(data)
View(data)


#count of diabetes and non-diabetes
library(ggplot2)


ggplot(data, aes(Outcome, fill = Outcome)) + 
  geom_bar() +
  theme_bw() +
  labs(title = "Diabetes Classification", x = "Diabetes") +
  theme(plot.title = element_text(hjust = 0.5))

#age vs diabetes
ggplot(data = data, aes(x = Age, fill = Outcome)) +
  geom_bar(stat='count', position='dodge') + ggtitle("Age Vs Diabetes") +
theme_bw() + labs(x = "Age") +theme(plot.title = element_text(hjust = 0.5)) 


#Pregnancies vs diabetes
ggplot(data = data, aes(x = Pregnancies, fill = Outcome)) +
  geom_bar(stat='count', position='dodge') + ggtitle("Pregnancies Vs Diabetes") +
  theme_bw() + labs(x = "Pregnancies") +theme(plot.title = element_text(hjust = 0.5)) 

#Glucose vs diabetes
ggplot(data = data, aes(x = Glucose, fill = Outcome)) +
  geom_bar(stat='count', position='dodge') + ggtitle("Glucose Vs Diabetes") +
  theme_bw() + labs(x = "Glucose") +theme(plot.title = element_text(hjust = 0.5)) 


#BloodPressure vs diabetes
ggplot(data = data, aes(x = BloodPressure, fill = Outcome)) +
  geom_bar(stat='count', position='dodge') + ggtitle("BP Vs Diabetes") +
  theme_bw() + labs(x = "BloodPressure") +theme(plot.title = element_text(hjust = 0.5)) 



#BMI vs diabetes
ggplot(data = data, aes(x = BMI, fill = Outcome)) +
  geom_bar(stat='count', position='dodge') + ggtitle("BMI Vs Diabetes") +
  theme_bw() + labs(x = "BMI") +theme(plot.title = element_text(hjust = 0.5)) 





#using ggplot to create a boxplot (data for distribution)
p1 <- ggplot(data, aes(x=Outcome , y=BMI, fill=Outcome))+geom_boxplot()
print(p1) 
#high Bmi usually means high risk of diabetes.




#using ggplot to create a boxplot (data for distribution) for another predictor
p2 <- ggplot(data, aes(x=Outcome , y=Glucose, fill=Outcome))+geom_boxplot()
print(p2)




#Relation between age and glucose
p3 <- ggplot(data, aes(x=Age , y=Glucose , col=Outcome))+geom_point()
p3



#boxplot for pregnancies and age
p4<-ggplot(data, aes(x=Pregnancies, y=Age, color=Outcome)) +geom_boxplot()
p4


#Relation between age and BMI
p5 <- ggplot(data, aes(x=Age , y=BMI , col=Outcome))+geom_point()
p5





#MACHINE LEARNING


#Comparing Multiple Models ACCURACY
#Train and Test Split


set.seed(123)
index <- sample(2, nrow(data), prob = c(0.8, 0.2), replace = TRUE)
Diabetes_train <- data[index==1, ] # Train data
View(Diabetes_train)
print(dim(Diabetes_train))
Diabetes_test <- data[index == 2, ] # Test data
View(Diabetes_test)
print(dim(Diabetes_test))



#Model Training
#Logistic Regression ----------------------------------------

Logistic_reg <- glm(formula = Outcome ~ ., family = binomial, 
                    data = Diabetes_train)
summary(Logistic_reg)




#Let's see how accurate our model is.
glm_pred <- predict(Logistic_reg,Diabetes_test,type= "response")
glm_pred



#The R function table() can be used to produce a confusion matrix 
accurate <- (table(ActualValue= Diabetes_test$Outcome,Result=glm_pred>0.5))
accurate
sum(diag(accurate)) / sum(accurate)
#We obtian an accuracy of 73.54%


lr_model <- (accurate[2, 2] + accurate[1, 1]) / sum(accurate)
lr_model




#Training a Decision Tree----------------------------------------------

# Train a decision tree model
library(rpart.plot)
Diabetes_model <- rpart(formula = Outcome ~., 
                        data = Diabetes_train, 
                        method = "class")   # method should be specified as the class for the classification task.

rpart.plot(x=Diabetes_model,yesno=2,type=0,extra=0)




#Model Performance Evaluation

#Next, step is to see how our trained model performs on the test/unseen dataset. For predicting the test data class we need to supply the model object, test dataset and the type = "class" inside the predict( ) function.

# class prediction
dt_pred <- predict(object = Diabetes_model,  
                           newdata = Diabetes_test,   
                           type = "class")
dt_pred
#(a) Confusion matrix

#To evaluate the test performance we are going to use the confusionMatrix( ) from caret library. We can observe that out of 155 observations it wrongly predicts (18+31=49) observations. The model has achieved about 68.39% accuracy using a single decision tree.
# Generate a confusion matrix for the test data
install.packages("caret")
library(caret)
confusionMatrix(data = dt_pred,       
                reference = Diabetes_test$Outcome)

dt_model<- confusionMatrix(dt_pred, Diabetes_test$Outcome)$overall['Accuracy']
dt_model






#Applying random forests model --------------
install.packages("randomForest")
library(randomForest)
set.seed(123)
rf_data <- randomForest(Outcome ~., data = Diabetes_train, 
                        mtry = 8, ntree=50, importance = TRUE)
rf_data
#we want to predict Outcome using each of the remaining columns of data.



# Testing the Model
rf_pred <- predict(rf_data, newdata = Diabetes_test)
library(caret)
confusionMatrix(data=rf_pred, reference=Diabetes_test$Outcome)
#The model achieved 69.67
rf_model <- confusionMatrix(rf_pred, Diabetes_test$Outcome)$overall['Accuracy']
rf_model


#Checking the variable importance
#Variable importance refers to how much a given model "uses" that variable to make accurate predictions.
importance(rf_data)


par(mfrow = c(1, 2))
varImpPlot(rf_data, type = 2, main = "Variable Importance",col = 'black')


#we notice that glucose,bmi,diabeticpedigree are important variable





#Applying Support Vector Machine - svm model --------
install.packages("e1071")
library(e1071)
svm_model  <- svm(Outcome ~., data = Diabetes_train, kernel = "radial",
                  gamma = 0.01, cost = 10) 
summary(svm_model)

#Testing the Model:
svm_pred <- predict(svm_model, newdata = Diabetes_test)
library(caret)
confusionMatrix(svm_pred, Diabetes_test$Outcome)

svm_model <- confusionMatrix(svm_pred, Diabetes_test$Outcome)$overall['Accuracy']
svm_model
#This model achieved 69.68% of accuracy

#Comparison of Model Accuracy 
library(ggplot2)
accuracy <- data.frame(Model=c("Logistic Regression","Decision Tree",
                               "Random Forest", "Support Vector Machine (SVM)"),
                       Accuracy <- c(lr_model,dt_model,rf_model,svm_model) )  
ggplot(accuracy,aes(x=Model,y=Accuracy)) + geom_bar(stat='identity') + theme_bw() + ggtitle('Comparison of Model Accuracy')









































































































