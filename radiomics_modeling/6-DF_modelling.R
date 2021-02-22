# author: Xia Wei, xiaw@sibet.ac.cn
# paper: Deep Learning for Automatic Differential Diagnosis of 
# Primary Central Nervous System Lymphoma and
# Glioblastoma: Multi-parametric MRI based Convolutional
# Neural Network Model
# date: 02/13/2021


# # DF radiomics model building and test
# combing the predictions of each single parametric radiomics model using logistic regression

# for each fold in CV


# DF_RADS model building
DF_rads_model_building <- function(label, T1C_preds, T2F_preds, ADC_preds){
  predicts_all_training <- cbind(label, T1C_preds, T2F_preds, ADC_preds)
  colnames_predicts_all <- c('label','T1C', 'T2F', 'ADC')
  colnames(predicts_all_training) <- colnames_predicts_all
  predicts_all_training<- as.data.frame(predicts_all_training)
  #
  formula_DF <- as.formula('label~T1C+T2F+ADC')
  
  # logistic regression for combining the predictions of single parametric radiomics models
  set.seed(123)
  glm.fit <- glm(formula_DF,
                 data=predicts_all_training,family=binomial(link="logit"))
  return(glm.fit)
}


DF_rads_model_test <- function(label, T1C_preds, T2F_preds, ADC_preds, DF_rads_model){
  predicts_all_test <- cbind(label, T1C_preds, T2F_preds, ADC_preds)
  colnames_predicts_all <- c('label','T1C', 'T2F', 'ADC')
  colnames(predicts_all_test) <- colnames_predicts_all
  predicts_all_test<- as.data.frame(predicts_all_test)
  
  test_predictions <- predict(DF_rads_model, predicts_all_test, type = 'response')
  
  # performance metric for rads model
  test_auc <- roc(as.factor(label),as.numeric(test_predictions),
                  plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
  test_auc_val <- test_auc$auc[1]
  Youden_index <- test_auc$sensitivities + test_auc$specificities
  # best_cutoff <-  test_auc$thresholds[Youden_index==max(Youden_index)]
  best_cutoff <- 0.5
  func_acc_sen_spe(label,test_predictions,best_cutoff,1)
  acc_sen_spe_auc <- c(func_acc_sen_spe(label,test_predictions,best_cutoff,1), test_auc_val)
  print(acc_sen_spe_auc)
  
  return(acc_sen_spe_auc)
  
}

# fold -- f1
DF_rads_model_f1 <- DF_rads_model_building(data_T1C_f1[[2]], 
                                           T1C_rads_model_f1_training_preds[[2]],
                                           T2F_rads_model_f1_training_preds[[2]],
                                           ADC_rads_model_f1_training_preds[[2]])

DF_rads_model_f1_metrics <- DF_rads_model_test(data_T1C_f1[[4]], 
                                               T1C_rads_model_f1_test_preds[[2]],
                                               T2F_rads_model_f1_test_preds[[2]],
                                               ADC_rads_model_f1_test_preds[[2]],
                                               DF_rads_model_f1)

# fold -- f2
DF_rads_model_f2 <- DF_rads_model_building(data_T1C_f2[[2]], 
                                           T1C_rads_model_f2_training_preds[[2]],
                                           T2F_rads_model_f2_training_preds[[2]],
                                           ADC_rads_model_f2_training_preds[[2]])

DF_rads_model_f2_metrics <- DF_rads_model_test(data_T1C_f2[[4]], 
                                               T1C_rads_model_f2_test_preds[[2]],
                                               T2F_rads_model_f2_test_preds[[2]],
                                               ADC_rads_model_f2_test_preds[[2]],
                                               DF_rads_model_f2)

# fold -- f3
DF_rads_model_f3 <- DF_rads_model_building(data_T1C_f3[[2]], 
                                           T1C_rads_model_f3_training_preds[[2]],
                                           T2F_rads_model_f3_training_preds[[2]],
                                           ADC_rads_model_f3_training_preds[[2]])

DF_rads_model_f3_metrics <- DF_rads_model_test(data_T1C_f3[[4]], 
                                               T1C_rads_model_f3_test_preds[[2]],
                                               T2F_rads_model_f3_test_preds[[2]],
                                               ADC_rads_model_f3_test_preds[[2]],
                                               DF_rads_model_f3)

# fold -- f4
DF_rads_model_f4 <- DF_rads_model_building(data_T1C_f4[[2]], 
                                           T1C_rads_model_f4_training_preds[[2]],
                                           T2F_rads_model_f4_training_preds[[2]],
                                           ADC_rads_model_f4_training_preds[[2]])

DF_rads_model_f4_metrics <- DF_rads_model_test(data_T1C_f4[[4]], 
                                               T1C_rads_model_f4_test_preds[[2]],
                                               T2F_rads_model_f4_test_preds[[2]],
                                               ADC_rads_model_f4_test_preds[[2]],
                                               DF_rads_model_f4)

# fold -- f5
DF_rads_model_f5 <- DF_rads_model_building(data_T1C_f5[[2]], 
                                           T1C_rads_model_f5_training_preds[[2]],
                                           T2F_rads_model_f5_training_preds[[2]],
                                           ADC_rads_model_f5_training_preds[[2]])

DF_rads_model_f5_metrics <- DF_rads_model_test(data_T1C_f5[[4]], 
                                               T1C_rads_model_f5_test_preds[[2]],
                                               T2F_rads_model_f5_test_preds[[2]],
                                               ADC_rads_model_f5_test_preds[[2]],
                                               DF_rads_model_f5)



# the mean metric of CV
DF_rads_CV_mean_metric <- (DF_rads_model_f1_metrics
                           +DF_rads_model_f2_metrics
                           +DF_rads_model_f3_metrics
                           +DF_rads_model_f4_metrics
                           +DF_rads_model_f5_metrics)/5
DF_rads_CV_mean_metric
