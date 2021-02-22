# author: Xia Wei, xiaw@sibet.ac.cn
# paper: Deep Learning for Automatic Differential Diagnosis of 
# Primary Central Nervous System Lymphoma and
# Glioblastoma: Multi-parametric MRI based Convolutional
# Neural Network Model
# date: 02/13/2021

# load radiomics features 


library(ggplot2)
library(glmnet)
library(caret)
library(pROC)
library(rms)
library(mRMRe)
# A--GBM B--PCNSL

# load data set split info
all_info <- read.csv('./CV_5folds_split.csv')

A_info <- all_info[all_info$label==0,]
A_info <- A_info[order(A_info[,1]),]
B_info <- all_info[all_info$label==1,]
B_info <- B_info[order(B_info[,1]),]

# load rad features
Fea_A_T1C <- read.csv('./rad_feas/rad_features_GBM_T1C.csv')     # GBM
Fea_B_T1C <- read.csv('./rad_feas/rad_features_PCNSL_T1C.csv')   # PCNSL

Fea_A_T2F <- read.csv('./rad_feas/rad_features_GBM_T2F.csv')     # GBM
Fea_B_T2F <- read.csv('./rad_feas/rad_features_PCNSL_T2F.csv')   # PCNSL

Fea_A_ADC <- read.csv('./rad_feas/rad_features_GBM_ADC.csv')     # GBM
Fea_B_ADC <- read.csv('./rad_feas/rad_features_PCNSL_ADC.csv')   # PCNSL

# exclude the patients did not meet the inclusion and exclusion criterias
# according to the clinical info file (CV_5folds_split.csv)
Fea_A_T1C <- Fea_A_T1C[which(Fea_A_T1C$id %in% A_info$id),]
Fea_A_T2F <- Fea_A_T2F[which(Fea_A_T2F$id %in% A_info$id),]
Fea_A_ADC <- Fea_A_ADC[which(Fea_A_ADC$id %in% A_info$id),]

Fea_B_T1C <- Fea_B_T1C[which(Fea_B_T1C$id %in% B_info$id),]
Fea_B_T2F <- Fea_B_T2F[which(Fea_B_T2F$id %in% B_info$id),]
Fea_B_ADC <- Fea_B_ADC[which(Fea_B_ADC$id %in% B_info$id),]

# all features combined
Fea_T1C <- rbind(Fea_A_T1C[,-1], Fea_B_T1C[,-1])
Fea_T2F <- rbind(Fea_A_T2F[,-1], Fea_B_T2F[,-1])
Fea_ADC <- rbind(Fea_A_ADC[,-1], Fea_B_ADC[,-1])

id <- all_info$id
label <- all_info$label

