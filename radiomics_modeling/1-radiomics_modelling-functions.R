# author: Xia Wei, xiaw@sibet.ac.cn
# paper: Deep Learning for Automatic Differential Diagnosis of 
# Primary Central Nervous System Lymphoma and
# Glioblastoma: Multi-parametric MRI based Convolutional
# Neural Network Model
# date: 02/13/2021


# functions for radiomics model training and test

# calculate the accuracy sensitivity and specificity
func_acc_sen_spe <- function(label,predictions,cutoff,pos_label){
  right_num <- 0
  pred <- 0
  TP <- 0
  FP <- 0
  TN <- 0
  FN <- 0
  if(pos_label==1){
    for(ind in seq(length=length(label),from = 1,to=length(label))){
      if(predictions[ind]>cutoff){pred <- 1}else{pred <- 0}
      if(label[ind]==1 & pred==1){TP <- TP +1 }
      if(label[ind]==0 & pred==1){FP <- FP +1 }
      if(label[ind]==0 & pred==0){TN <- TN +1 }
      if(label[ind]==1 & pred==0){FN <- FN +1 }
    }
  }
  
  if(pos_label==0){
    for(ind in seq(length=length(label),from = 1,to=length(label))){
      if(predictions[ind]<cutoff){pred <- 1}else{pred <- 0}
      if(label[ind]==1 & pred==1){TP <- TP +1 }
      if(label[ind]==0 & pred==1){FP <- FP +1 }
      if(label[ind]==0 & pred==0){TN <- TN +1 }
      if(label[ind]==1 & pred==0){FN <- FN +1 }
    }
  }
  right_num <- TP + TN
  accuracy <- right_num/length(label)
  sensitivity <- TP/(TP+FN)
  specificity <- TN/(TN+FP)
  
  # precision <- TP/(TP+FP)
  # recall <- TP/(TP+FN)
  # 
  # f1 = (2*precision*recall)/(precision+recall)
  return(rbind(accuracy, sensitivity, specificity))
}

# 
# test model performance
radiomics_model_test <- function(rad_model, fea_names, x_test, y_test){
  coef_model<- coef(rad_model$glmnet.fit,s <- rad_model$lambda.1se)
  model_feas_names <- coef_model@Dimnames[[1]][coef_model@i+1]
  x_test <- x_test[,which(colnames(x_test) %in% fea_names)]

  # performance metric for rads model
  test_predictions <- predict(rad_model$glmnet.fit, as.matrix(x_test), s= rad_model$lambda.1se, type = 'response')
  test_auc <- roc(as.factor(y_test),as.numeric(test_predictions),
                  plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
  test_auc_val <- test_auc$auc[1]
  Youden_index <- test_auc$sensitivities + test_auc$specificities
  # best_cutoff <-  test_auc$thresholds[Youden_index==max(Youden_index)]
  best_cutoff<-0.5
  func_acc_sen_spe(y_test,test_predictions,best_cutoff,1)
  acc_sen_spe_auc <- c(func_acc_sen_spe(y_test,test_predictions,best_cutoff,1), test_auc_val)
  print(acc_sen_spe_auc)

  return(list(acc_sen_spe_auc, test_predictions))
}


# for radiomics model training
# obtain model and feature names for test
radiomics_model_training <- function(x_train, y_train){
  # feature selection based on spearman correlation matrix
  # Determine highly correlated variables based on training set
  # exclude the highly correlated features
  Fea.cor <- cor(x_train, use = "everything",method = "spearman")
  num <- findCorrelation(Fea.cor, cutoff = 0.9, verbose = FALSE, names = FALSE, exact = FALSE)
  x_train <- x_train[,-sort(num)]
  
  # mrmr feature selcetion -- accroding to the 10 times law
  fea_counts <- round(nrow(y_train)/10)
  mrdata_train_nor <- cbind(y_train,as.matrix(x_train))
  mrdata <- mRMR.data(data = data.frame(mrdata_train_nor) )
  mrlist <- mRMR.classic(data = mrdata,target_indices = c(1),method ='exhaustive',continuous_estimator = 'spearman',
                         feature_count = fea_counts)
  x_train <- subset(x_train,select = c(mrlist@feature_names[sort(mrlist@filters$'1')]))

  
  # LASSO modelling
  set.seed(123)
  cvfit <- cv.glmnet(as.matrix(x_train),y_train,
                         nfolds=10,family = "binomial",type.measure="auc")
  s_val <- cvfit$lambda.1se
  
  res <- list(cvfit, colnames(x_train))
  
  return(res)
  
  # # model details
  # coef_cvfit<- coef(cvfit$glmnet.fit,s <- cvfit$lambda.1se)
  # rads_model_feas_names <- coef_cvfit@Dimnames[[1]][coef_cvfit@i+1]
  # rads_model_coefs <- coef_cvfit@x
  # rads_model_feas_names
  # rads_model_coefs
  
  # # print models
  # rads_model <-  c()
  # for(ind in c(1:length(coef_cvfit@x))){
  #   rads_model <- paste(rads_model, '+',rads_model_coefs[ind],'*',rads_model_feas_names[ind])
  # }
  # print(rads_model)
}


# split training and test set with 5 fold CV
# the same with deep learning model training and test
training_test_set_split <- function(label, Feas, set_fold){
  allin_dataset <- cbind(label, set_fold, Feas) # specify the MRI sequence radiomics features
  
  training_dataset <- allin_dataset[allin_dataset$set=='train'|allin_dataset$set=='val',]
  test_dataset <- allin_dataset[allin_dataset$set=='test',]
  
  y_train <- as.matrix(training_dataset$label)
  y_test <- as.matrix(test_dataset$label)
  
  x_train <- training_dataset[,-(1:3)]
  x_test <- test_dataset[,-(1:3)]
  
  return(list(x_train, y_train, x_test, y_test))
}



