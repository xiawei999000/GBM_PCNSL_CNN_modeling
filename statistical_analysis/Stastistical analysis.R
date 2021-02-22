# author: Xia Wei, xiaw@sibet.ac.cn
# paper: Deep Learning for Automatic Differential Diagnosis of 
# Primary Central Nervous System Lymphoma and
# Glioblastoma: Multi-parametric MRI based Convolutional
# Neural Network Model
# date: 02/13/2021


# for stastical test
library(ggplot2)
library(glmnet)
library(caret)
library(pROC)
library(rms)
library(mRMRe)




# stastastic analysis for clincal information
info_all <- read.csv('./ClincalInfo&DiagnosisByRadiologists.csv')

# for age using U-test
mean(info_all$age[which(info_all$label==0)])
sd(info_all$age[which(info_all$label==0)])

mean(info_all$age[which(info_all$label==1)])
sd(info_all$age[which(info_all$label==1)])

mean(info_all$age)
sd(info_all$age)

wilcox.test(info_all$age[which(info_all$label==0)], info_all$age[which(info_all$label==1)], exact = F)

# for gender using fisher exact test
c1 <- length(which(info_all$label==0&info_all$sex==1))
c2 <- length(which(info_all$label==0&info_all$sex==0))
c3 <- length(which(info_all$label==1&info_all$sex==1))
c4 <- length(which(info_all$label==1&info_all$sex==0))

c <- matrix(c(c1, c2, c3, c4), nrow = 2)

fisher.test(c)


# for calculation of the performance of radiologists
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

metrics_calc_preds_label <- function(preds, label){
  auc_roc <- roc(as.factor(label),as.numeric(preds), 
                 plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
  auc <- auc_roc$auc[1]
  # ci(test_auc_all_cross)
  # test performance 
  Youden_index <- auc_roc$sensitivities + auc_roc$specificities
  best_cutoff <-  auc_roc$thresholds[Youden_index==max(Youden_index)]
  acc_sen_spe <- func_acc_sen_spe(label,preds,best_cutoff,1)
  acc_sen_spe_auc <- c(acc_sen_spe, auc)
  return(acc_sen_spe_auc)
}


# performance metrics of radiologists
metrics_calc_preds_label(info_all$doctor_diagnosis_5, info_all$label)
metrics_calc_preds_label(info_all$doctor_diagnosis_10, info_all$label)
metrics_calc_preds_label(info_all$doctor_diagnosis_20, info_all$label)



# compare the acurracy between models and radiologists using Pearson's Chi-squared Test
total_num=289
acc1=0.899
acc2=0.906
# prop.test(x = c(acc1*total_num, acc2*total_num), n = c(total_num, total_num))

c1=acc1*total_num
c2=total_num - c1
c3=acc2*total_num
c4=total_num - c3
c <- matrix(c(c1, c2, c3, c4), nrow = 2)
chisq.test(c)

