## set the working directory path
setwd("E:/Data science/Edvancer/Business_Predictive Analytics in R module/Project Submission__Business_Predictive Analytics in R_Naidu/Project_2_Retail/Prject Data_2")
# current working directory
getwd()

# Reading the data
store_train=read.csv("store_train.csv")
store_test=read.csv("store_test.csv")

# importing the libraries
library(dplyr)

# data
glimpse(store_train)
glimpse(store_test)

# function for creating dummies based on frequency
CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    # removing or converting symbols from the dummie column name
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    # converting into numeric 
    data[,name]=as.numeric(data[,var]==cat)
  }
  # 
  data[,var]=NULL
  return(data)
}


# creating new column:"store" and "data" to merge the train and test data
store_test$store=NA
# column:"data" created For separating train and test data after data processing
store_train$data='train'
store_test$data='test'

# combining the train and test data for data pre processing
store_all=rbind(store_train,store_test)

#-------------------------------------------------------------------------------
# converting all categorical variables as categorical variables------------------
glimpse(store_all)
store_all$country=as.character(store_all$country)
store_all$State=as.character(store_all$State)

# no of unique values in each column
lapply(store_all,function(x) length(unique(x)))
# filling the na values with median value
store_all[(is.na(store_all$country)),"country"]=median(as.numeric(store_train$country),na.rm = T)

# removing the some columns whos information is already capured in other columns
store_all_1=store_all %>% select(-countyname,-storecode,-Areaname,-countytownname,
                                 -State,)
# creating dummies for all categorical-----------------------------------------
glimpse(store_all_1)

# names of categorical columns 
names(store_all_1)[sapply(store_all_1,function(x) is.character(x))]

cat_cols=c("country","state_alpha","store_Type")

# creating dummies based on frequency
for(cat in cat_cols){
  store_all_1=CreateDummies(store_all_1,cat,40)
}

## identifying the missing values------------------------------------------------
glimpse(store_all_1)
lapply(store_all_1,function(x) sum(is.na(x)))

# missing values present in "population" are replaced with mean/median values
store_all_1[is.na(store_all_1$population),"population"]= mean(store_all_1$population,na.rm = T)

##-----------------------------------------------------------------------------
## dividing data into train & test data-----------------------------------------
# filtering the train and test data and removing the columns created earlier for combining
store_train=store_all_1 %>% filter(data=="train") %>% select(-data)
store_test=store_all_1 %>% filter(data=="test") %>% select(-data,-store)

# for reproducing the same sample records for train and test data 
set.seed(2)
s=sample(1:nrow(store_train),0.8*nrow(store_train))

store_train1=store_train[s,]
store_train2=store_train[-s,]
##------------------------------------------------------------------------------
##To examine VIF(Variance inflation factor) to identify multy co-linearity, we can run a linear regression.
# importing the library
library(car)

# linear regression model on train data excluding the "Id" column
for_vif=lm(store~.-Id,data=store_train1)

# selecting top 5 features(VIF<=5)
sort(vif(for_vif),decreasing = T)[1:5]

# iteration-1(removing sales0)
for_vif=lm(store~.-Id-sales0,data=store_train1)
sort(vif(for_vif),decreasing = T)[1:5]

# iteration-2(removing sales2) 
for_vif=lm(store~.-Id-sales0-sales2,data=store_train1)
sort(vif(for_vif),decreasing = T)[1:5]

# iteration-3(removing sales3)
for_vif=lm(store~.-Id-sales0-sales2-sales3,data=store_train1)
sort(vif(for_vif),decreasing = T)[1:5]

#iteration-4 (removing sales1)
for_vif=lm(store~.-Id-sales0-sales2-sales3-sales1,data=store_train1)
sort(vif(for_vif),decreasing = T)[1:5]

##-----------------------------------------------------------------------------
# building logistic regression model on train data after removing multi-collinearity ---
store_log_fit=glm(store~.-Id-sales0-sales2-sales3-sales1,data=store_train1,family ="binomial")
store_log_fit=step(store_log_fit)

formula(store_log_fit)
store_log_fit=glm(store ~ sales4 + CouSub + population + country_31 + country_29 + 
                    country_13 + country_19 + country_15 + country_11 + country_9 + 
                    state_alpha_WV + state_alpha_CA + state_alpha_CO + state_alpha_LA + 
                    state_alpha_PR + state_alpha_IN + state_alpha_TN + state_alpha_KY + 
                    state_alpha_GA + state_alpha_VT + state_alpha_NH + state_alpha_MA
                  -state_alpha_KY-CouSub-country_29,
                  data=store_train1,family ="binomial")

summary(store_log_fit)
## from here we can drop vars one by one which had higher p-value
# store_log_fit=glm(store ~ sales4 + population + state_alpha_GA + state_alpha_CT + 
#                     state_alpha_VT + state_alpha_NH + state_alpha_MA-state_alpha_GA,
#                   data=store_train1,family ="binomial")

# summary(store_log_fit)

## performance of score of logistic regression model on validation data---------
library(pROC)
# predictions and AUC-ROC score on train(train1) data 
train.score=predict(store_log_fit,newdata = store_train1,type='response')
train_auc_score=auc(roc(store_train2$store,val.score))
train_auc_score
#Area under the curve: 0.7703

# predictions and AUC-ROC score on validation(train2) data
val.score=predict(store_log_fit,newdata = store_train2,type='response')
val_auc_score=auc(roc(store_train2$store,val.score))
val_auc_score

#Area under the curve: 0.7703
## model is stable as performance of model on train and validation data set is same

#-----------------------------------------------------------------------------
library(ggplot2)
mydata=data.frame(store=as.factor(store_train2$store),val.score=val.score)

# ggplot of predicted probability vs actual values
ggplot(mydata,aes(y=store,x=val.score,color=factor(store)))+
  geom_point()+geom_jitter()

###-----------------------------------------------------------------------------
## multi colinearity on total train data----------------------------------------
for_vif=lm(store~.-Id-sales0-sales2-sales3-sales1,data=store_train)

sort(vif(for_vif),decreasing = T)[1:5]

# now lets build the model on entire training data-----------------------------
store_log_fit_final=glm(store~.-Id-sales0-sales2-sales3-sales1,data=store_train,family ="binomial")
store_log_fit_final=step(store_log_fit_final)

formula(store_log_fit_final)
store_log_fit_final=glm(store ~ sales4 + CouSub + population + country_31 + country_29 + 
                          country_19 + country_15 + country_11 + country_9 + state_alpha_SC + 
                          state_alpha_WV + state_alpha_NY + state_alpha_CO + state_alpha_LA + 
                          state_alpha_SD + state_alpha_AL + state_alpha_FL + state_alpha_PA + 
                          state_alpha_WI + state_alpha_AR + state_alpha_OK + state_alpha_PR + 
                          state_alpha_MS + state_alpha_MI + state_alpha_OH + state_alpha_IN + 
                          state_alpha_NE + state_alpha_TN + state_alpha_IA + state_alpha_IL + 
                          state_alpha_KS + state_alpha_MO + state_alpha_KY + state_alpha_VA + 
                          state_alpha_GA + state_alpha_TX + state_alpha_VT + state_alpha_NH + 
                          state_alpha_MA-country_29-state_alpha_NE-state_alpha_SD-state_alpha_VA
                        -state_alpha_FL-state_alpha_CO-state_alpha_SC-state_alpha_NY-state_alpha_IA,
                        data=store_train,family ="binomial")

summary(store_log_fit_final)

# library(pROC)
# 
val_score_final=predict(store_log_fit_final,newdata = store_train2,type='response')
auc_score_final=auc(roc(store_train2$store,val_score_final))
auc_score_final


##---------------------------------------------------------------------------------
##--------Decision Tree Model---------------------------------------------------
#---------------------------------------------------------------
# converting store 0/1 into factor/character in store_train1
store_train1$store=as.factor(store_train1$store) 
store_train$store=as.factor(store_train$store)
library(tree)
tree.store=tree(store~.-Id,data=store_train1)
tree.store
summary(tree.store)

## performance of score of DT model on train1 and validation data(train2)---------
library(pROC)
tree.score.train1=predict(tree.store,newdata=store_train1,type="vector")[,2]

auc_score=auc(roc(store_train1$store,tree.score.train1))
auc_score
#Area under the curve: 0.7597

#-------------------------------------------------------------
tree.score.train2=predict(tree.store,newdata=store_train2,type="vector")[,1]

auc_score=auc(roc(store_train2$store,tree.score.train2))
auc_score

#Area under the curve: 0.7283
#------------------------------------------------------------------------------
plot(tree.store)
text(tree.store,pretty=0, cex=0.7)

#-------------------Pruning-----------------------------------------------------
set.seed(3)
cv.tree.store=cv.tree(tree.store, FUN=prune.misclass)
plot(cv.tree.store$size,cv.tree.store$dev,type='b')
# minimum deviance occurs at full tree model. so no scope for improvement.
## ------------------------------------------------------------------------
prune.tree.store=prune.misclass(tree.store,best=7)

plot(prune.tree.store)
text(prune.tree.store,pretty=0)

prune.tree.store

## ------------------------------------------------------------------------
##------------------------------------------------------------------------------
#-------------------randomForest-------------------------------------------------

library(randomForest)

class_rf=randomForest(store~.-Id,data=store_train1)

class_rf
summary(class_rf)

## ------------------------------------------------------------------------
forest.pred1=predict(class_rf,newdata=store_train1,type = "prob")[,1]
auc_score1=auc(roc(store_train1$store,forest.pred1))
auc_score1

forest.pred2=predict(class_rf,newdata=store_train2,type = "prob")[,1]
auc_score2=auc(roc(store_train2$store,forest.pred2))
auc_score2
#----------RF model on entire train data---------------------------------------
class_rf_final=randomForest(store~.-Id,data=store_train,do.trace=T)
class_rf_final

# prediction on test data----------------------------------------------------
forest.pred.final=predict(class_rf_final,newdata = store_test,type="prob")[,2]
write.csv(forest.pred.final,"NaiduBabu_Yadla_P2_part2.csv",row.names=F)

#------------------------------------------------------------------------------
imp = data.frame(importance(class_rf_final)) 
# Classification RF model measures MeanDecreaseGini to identify 
# the importance of variables
imp
imp[order(imp[,1], decreasing = T),]

varImpPlot(class_rf_final)





