# GBM_PCNSL_CNN_modeling
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model
![The schematic workflow of CNN models development. The first step consisted of a series of automatic image preprocessing procedures, including image registration, brain extraction, and standardization. After image preprocessing, DenseNet based CNN models were developed by transfer learning, and a single-parametric CNN model was built using CE-T1WI, FLAIR, and ADC, respectively. A multi-parametric CNN model was developed using image-level fusion and decision-level fusion, respectively.](https://github.com/xiawei999000/GBM_PCNSL_CNN_modeling/blob/main/Figure%201.tif)


# code structure
> [CNN_modeling/](https://github.com/xiawei999000/GBM_PCNSL_CNN_modeling/tree/main/CNN_modeling): CNN model training, test and a few sample images.
> >[CNN_training_program/](https://github.com/xiawei999000/GBM_PCNSL_CNN_modeling/tree/main/CNN_modeling/CNN_training_program): CNN model training, heat map generation, and a link for sample images.
> >[CNN_model_test_and_statistical_analysis/](https://github.com/xiawei999000/GBM_PCNSL_CNN_modeling/tree/main/CNN_modeling/CNN_model_test_and_statistical_analysis): CNN model test and performance metrics calculation.  

> [radiomics_modeling/](https://github.com/xiawei999000/GBM_PCNSL_CNN_modeling/tree/main/radiomics_modeling): radiomics features extraction, feature selection, and model building.

> [preprocessing/](https://github.com/xiawei999000/GBM_PCNSL_CNN_modeling/tree/main/preprocessing): Executable programs and scripts for image preprocessing (DicomSeriesReadImageWrite: dicom series read and volume write, bet2: brain mask extraction, Resampler: resampling to the reference volume).  

> [statistical_analysis/](https://github.com/xiawei999000/GBM_PCNSL_CNN_modeling/tree/main/statistical_analysis): Radiologists' performance metrics calculation and statistical analysis.
