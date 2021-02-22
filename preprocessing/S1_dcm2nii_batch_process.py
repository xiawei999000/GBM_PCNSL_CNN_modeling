'''
@File        : S1_dcm2nii_batch_process.py
@Author      : xiaw@sibet.ac.cn
@Date        : 2021/02/12
@Description :
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and
Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model

## script for transforming dcm series images to nii volume. (original dcm images to nii volume)
'''


import subprocess
import os

if __name__ == '__main__':
    DCM_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\PCNSL\\final\\PCNSL-dcm\\'
    nii_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\PCNSL\\standard\\PCNSL-nii\\'
    exe_path = './preprocesser/DicomSeriesReadImageWrite/DicomSeriesReadImageWrite.exe'

    pid_list = os.listdir(DCM_path)
    pid_empty_T1C = []
    pid_empty_T2F = []
    pid_empty_ADC = []
    pid_empty_DWI = []

    for idx in pid_list:

        if not os.path.exists(nii_path + str(idx)):
            os.mkdir(nii_path + str(idx))

        pid_dir_T1C = DCM_path + str(idx) + '\\T1C'
        pid_dir_T2F = DCM_path + str(idx) + '\\T2F'
        pid_dir_ADC = DCM_path + str(idx) + '\\ADC'
        pid_dir_DWI_L = DCM_path + str(idx) + '\\DWI_sort\\DWI_L'
        pid_dir_DWI_H = DCM_path + str(idx) + '\\DWI_sort\\DWI_H'

        new_dir_T1C = nii_path + str(idx) + '\\T1C.nii.gz'
        new_dir_T2F = nii_path + str(idx) + '\\T2F.nii.gz'
        new_dir_ADC = nii_path + str(idx) + '\\ADC.nii.gz'
        new_dir_DWI_L = nii_path + str(idx) + '\\DWI_L.nii.gz'
        new_dir_DWI_H = nii_path + str(idx) + '\\DWI_H.nii.gz'

        if not os.path.exists(new_dir_T1C):
            if os.path.exists(pid_dir_T1C):
                if len(os.listdir(pid_dir_T1C)) != 0:
                    cmd = [exe_path, pid_dir_T1C, new_dir_T1C]
                    subprocess.call(cmd)
                    print('patient ', str(idx), 'T1C transformed.')
                else:
                    pid_empty_T1C.append(str(idx))
            else:
                pid_empty_T1C.append(str(idx))

        if not os.path.exists(new_dir_T2F):
            if os.path.exists(pid_dir_T2F):
                if os.listdir(pid_dir_T2F):
                    cmd = [exe_path, pid_dir_T2F, new_dir_T2F]
                    subprocess.call(cmd)
                    print('patient ', str(idx), 'T2F transformed.')
                else:
                    pid_empty_T2F.append(str(idx))
            else:
                pid_empty_T2F.append(str(idx))

        if not os.path.exists(new_dir_ADC):
            if os.path.exists(pid_dir_ADC):
                if os.listdir(pid_dir_ADC):
                    cmd = [exe_path, pid_dir_ADC, new_dir_ADC]
                    subprocess.call(cmd)
                    print('patient ', str(idx), 'ADC transformed.')
                else:
                    pid_empty_ADC.append(str(idx))
            else:
                pid_empty_ADC.append(str(idx))

        if not os.path.exists(new_dir_DWI_L):
            if os.path.exists(pid_dir_DWI_L):
                if os.listdir(pid_dir_DWI_L):
                    cmd = [exe_path, pid_dir_DWI_L, new_dir_DWI_L]
                    subprocess.call(cmd)
                    print('patient ', str(idx), 'DWI_L transformed.')
                else:
                    pid_empty_DWI.append(str(idx))
            else:
                pid_empty_DWI.append(str(idx))

        if not os.path.exists(new_dir_DWI_H):
            if os.path.exists(pid_dir_DWI_H):
                if os.listdir(pid_dir_DWI_H):
                    cmd = [exe_path, pid_dir_DWI_H, new_dir_DWI_H]
                    subprocess.call(cmd)
                    print('patient ', str(idx), 'DWI_H transformed.')
                else:
                    pid_empty_DWI.append(str(idx))
            else:
                pid_empty_DWI.append(str(idx))

    print('pid without T1C : ', pid_empty_T1C)
    print('pid without T2F : ', pid_empty_T2F)
    print('pid without ADC : ', pid_empty_ADC)
    print('pid without DWI : ', pid_empty_DWI)
