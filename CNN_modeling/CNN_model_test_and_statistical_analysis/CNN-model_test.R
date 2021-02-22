# author: Xia Wei, xiaw@sibet.ac.cn
# paper: Deep Learning for Automatic Differential Diagnosis of 
# Primary Central Nervous System Lymphoma and
# Glioblastoma: Multi-parametric MRI based Convolutional
# Neural Network Model
# date: 02/13/2021

# load CNN predictions and calculate evaluation metrics
library(ggplot2)
library(glmnet)
library(caret)
library(pROC)
library(rms)
library(mRMRe)

func_acc_sen_spe_f1 <- function(label,predictions,cutoff,pos_label){
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

Obtain_P_level_predictions <- function(slice_prediction_list, tumor_type){
  
  Tumor_slice_preds<-slice_prediction_list[which(slice_prediction_list$V3==tumor_type),]
  p_ind_Tumor <- sort(unique(Tumor_slice_preds$V1))
  p_ind_preds_Tumor <- as.numeric((1:length(p_ind_Tumor)) * 0)
  
  label <- as.numeric((1:length(p_ind_Tumor)) * 0)
  if(tumor_type == 'PCNSL'){
    label <- label + 1
  }
  
  p_ind_preds_Tumor <- as.numeric((1:length(p_ind_Tumor)) * 0)
  for(ind in 1:length(p_ind_Tumor)){
    p_ind_temp <- p_ind_Tumor[ind]
    Tumor_slice_preds_p_ind <- Tumor_slice_preds[which(Tumor_slice_preds$V1==p_ind_temp),]
    p_ind_preds_Tumor[ind] <- mean(Tumor_slice_preds_p_ind$V4)
  }
  
  p_preds <- as.data.frame(cbind(p_ind_Tumor, label, p_ind_preds_Tumor))
  
  return(p_preds)
}

Obtain_P_level_predictions_all <- function(slice_prediction_list){
  
  fold_preds_GBM <-  Obtain_P_level_predictions(slice_prediction_list, 'GBM')
  fold_preds_PCNSL <-  Obtain_P_level_predictions(slice_prediction_list, 'PCNSL')
  fold_preds <- rbind(fold_preds_GBM, fold_preds_PCNSL)
  return(fold_preds)
  
}

metrics_calc <- function(slice_prediction_list){
  p_preds <-  Obtain_P_level_predictions_all(slice_prediction_list)
  # test auc
  auc_roc <- roc(as.factor(p_preds$label),as.numeric(p_preds$p_ind_preds_Tumor), 
                            plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
  auc <- auc_roc$auc[1]
  # ci(test_auc_all_cross)
  # test performance 
  Youden_index <- auc_roc$sensitivities + auc_roc$specificities
  # best_cutoff <-  auc_roc$thresholds[Youden_index==max(Youden_index)]
  best_cutoff <- 0.5
  acc_sen_spe <- func_acc_sen_spe_f1(p_preds$label,p_preds$p_ind_preds_Tumor,best_cutoff,1)
  acc_sen_spe_auc <- c(acc_sen_spe, auc)
  print(acc_sen_spe_auc)
  return(acc_sen_spe_auc)
}

metrics_calc_preds_label <- function(preds, label){
  auc_roc <- roc(as.factor(label),as.numeric(preds), 
                 plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
  auc <- auc_roc$auc[1]
  # ci(test_auc_all_cross)
  # test performance 
  Youden_index <- auc_roc$sensitivities + auc_roc$specificities
  # best_cutoff <-  auc_roc$thresholds[Youden_index==max(Youden_index)]
  best_cutoff <- 0.5
  acc_sen_spe <- func_acc_sen_spe_f1(label,preds,best_cutoff,1)
  acc_sen_spe_auc <- c(acc_sen_spe, auc)
  return(acc_sen_spe_auc)
}


mean_metrics_CV <- function(modal, fold_num, preds_path, procedure){
  
  fold_preds_metrics_all <- 0
  
  for(fold_ind in 1:fold_num){
    fold_temp_preds_path <- paste0(preds_path, '/' ,as.character(fold_num), '_fold_', as.character(fold_ind), '_', 
                                  modal,'_',procedure,'.txt', collapse = NULL)
    fold_temp_preds<-read.table(fold_temp_preds_path, header=F,na.strings = c("NA"))
    fold_temp_preds_metrics <- metrics_calc(fold_temp_preds)
    fold_preds_metrics_all <- fold_preds_metrics_all + fold_temp_preds_metrics
  }
  
  fold_preds_metrics_all_mean <- fold_preds_metrics_all/fold_num
  return(fold_preds_metrics_all_mean)
}


obtain_modal_preds_train_set <- function(modal, preds_path, fold_num, fold_ind){

  fold_preds_train_path <- paste0(preds_path, '/' ,as.character(fold_num), '_fold_', as.character(fold_ind), 
                                  '_', modal,'_train.txt', collapse = NULL)
  fold_preds_train<-read.table(fold_preds_train_path, header=F,na.strings = c("NA"))
  
  fold_preds_val_path <- paste0(preds_path, '/' ,as.character(fold_num), '_fold_', as.character(fold_ind), 
                                '_', modal,'_val.txt', collapse = NULL)
  fold_preds_val<-read.table(fold_preds_val_path, header=F,na.strings = c("NA"))
  
  p_preds_train <-  Obtain_P_level_predictions_all(fold_preds_train)
  p_preds_val <-  Obtain_P_level_predictions_all(fold_preds_val)

  p_preds_train_set <- rbind(p_preds_train, p_preds_val)
  # colnames(p_preds_train_set)[3] <- paste0(modal, '_preds')
  
  return(p_preds_train_set)
}

obtain_modal_preds_test_set <- function(modal, preds_path, fold_num, fold_ind){

  fold_preds_test_path <- paste0(preds_path, '/' ,as.character(fold_num), '_fold_', as.character(fold_ind), 
                                 '_', modal,'_test.txt', collapse = NULL)
  fold_preds_test<-read.table(fold_preds_test_path, header=F,na.strings = c("NA"))

  p_preds_test <-  Obtain_P_level_predictions_all(fold_preds_test)
  
  # colnames(p_preds_test_set)[3] <- paste0(modal, '_preds')
  
  return(p_preds_test)
}


DF_CNN_LR_test <- function(fold_num, preds_path){
  fold_preds_metrics_all <- 0
  
  for(fold_ind in 1:fold_num){
    T1C_training_set <- obtain_modal_preds_train_set('T1C', preds_path, fold_num, fold_ind)
    T2F_training_set <- obtain_modal_preds_train_set('T2F', preds_path, fold_num, fold_ind)
    ADC_training_set <- obtain_modal_preds_train_set('ADC', preds_path, fold_num, fold_ind)
    DF_training_set <- cbind(T1C_training_set$label, 
                             T1C_training_set$p_ind_preds_Tumor,
                             T2F_training_set$p_ind_preds_Tumor,
                             ADC_training_set$p_ind_preds_Tumor)
    DF_training_set <- as.data.frame(DF_training_set)
    colnames(DF_training_set) <- c('label', 'T1C_preds', 'T2F_preds', 'ADC_preds')
    # build DF-CNN model using logistic regression by linear combing of multi-parametric predictions
    formula_DF <- as.formula('label~T1C_preds+T2F_preds+ADC_preds')
    
    # multivariable logistic regression -- DF
    glm.fit_DF <- glm(formula_DF,
                      data=DF_training_set,family=binomial(link="logit"))
    
    # load test set for this fold
    T1C_test_set <- obtain_modal_preds_test_set('T1C', preds_path, fold_num, fold_ind)
    T2F_test_set <- obtain_modal_preds_test_set('T2F', preds_path, fold_num, fold_ind)
    ADC_test_set <- obtain_modal_preds_test_set('ADC', preds_path, fold_num, fold_ind)
    DF_test_set <- cbind(T1C_test_set$label, 
                         T1C_test_set$p_ind_preds_Tumor,
                         T2F_test_set$p_ind_preds_Tumor,
                         ADC_test_set$p_ind_preds_Tumor)
    DF_test_set <- as.data.frame(DF_test_set)
    colnames(DF_test_set) <- c('label', 'T1C_preds', 'T2F_preds', 'ADC_preds')
    
    # predictions of samples in test set
    predictions_DF_test <- predict(glm.fit_DF, DF_test_set, type = 'response')

    fold_temp_preds_metrics <- metrics_calc_preds_label(predictions_DF_test, DF_test_set$label)
    fold_preds_metrics_all <- fold_preds_metrics_all + fold_temp_preds_metrics
    print(fold_temp_preds_metrics)
  }
  
  fold_preds_metrics_all_mean <- fold_preds_metrics_all/fold_num
  return(fold_preds_metrics_all_mean)
}


# 5 CV performance metrics of T1C_CNN model
CV_5_folds_metrics_T1C_CNN <- mean_metrics_CV('T1C', 5, './CNN_CV_predictions', 'test')
CV_5_folds_metrics_T1C_CNN

# 5 CV performance metrics of T2F_CNN model
CV_5_folds_metrics_T2F_CNN <- mean_metrics_CV('T2F', 5, './CNN_CV_predictions', 'test')
CV_5_folds_metrics_T2F_CNN

# 5 CV performance metrics of ADC_CNN model
CV_5_folds_metrics_ADC_CNN <- mean_metrics_CV('ADC', 5, './CNN_CV_predictions', 'test')
CV_5_folds_metrics_ADC_CNN

# 5 CV performance metrics of IF_CNN model
CV_5_folds_metrics_IF_CNN <- mean_metrics_CV('cross_modal', 5, './CNN_CV_predictions', 'test')
CV_5_folds_metrics_IF_CNN

# 5 CV performance metrics of DF_CNN model
DF_CNN_LR_test(5, './CNN_CV_predictions')
