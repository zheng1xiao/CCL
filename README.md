#CCL

## Project Description
This replication package contains the dataset and code for our paper `Making SZZ on Just-in-Time Defect Prediction Robust to Mislabeled Changes`.  
The work proposes the CCL model to address the impact of incorrectly labeled code changes by SZZ in the domain of Just-In-Time defect prediction. The CCL model consists of a data denoising component and a joint learning component to reduce noise and prevent overfitting during model training. The study employs five comprehensive experiments to demonstrate the effectiveness of CCL in the field of Just-In-Time defect prediction.

## Environments

Language: Python (v3.8)

Python packages:
   * [cleanlab 0.1.1]   
   * [scikit-learn 0.24.1]
   * [PyTorch 1.12.1+cu116]
   * [imbalanced-learn 0.7.0]

We used the noise label classification and imbalanced classification sampling methods proposed by Xia et al. in our baselines, including four sampling methods: IF, OSS, RSDS, CLNI, GBS, and E2SC. Most of their work is available through replication package links, and we directly used their implementations. All baseline experiments are conducted in the "CCL_baseline" folder.

## File Organization
There are five folders.A folder represents an experiment.

Directories  
File configuration in the CCL folder  
CCL.py The main code of the CCL model  
loder.py The load data set is encapsulated as batch  
loss.py Dual network loss function in joint learning component  
modelMLP.py Network model in joint learning component  
dataset/:Store the data set needed for the experiment  

CCL_baseline/:Store the baseline and experimental code of CCL  
   CCL\：CCL model  
   IF\：Isolation Forest（IF）as a baseline  
   OSS\：One-Sided Selection（OSS）as a baseline  
   CLNI\：Closest List Noise Identification (CLNI) as a baseline  
   GBS\：Granular Ball Sampling（GBS）as a baseline  
   RSDS\：Random Space Division Sampling（RSDS）as a baseline  
   E2SC\：An Effective, Efficient, and Scalable Confidence-Based Instance Selection（E2SC）as a baseline  

CCL_structure/:Experimental code of RQ2  
   CCL.py：CCL model  
   c1.py：Single network structure  
   CT.py：Only the joint learning component  
   CL.py：Only the data denoising component  
   
CCL_classifier/:Experimental code of RQ3  
   ORI-LR.py：The CCL model uses the LR as a network  
   ORI-MLP.py：The CCL model uses the MLP as a network  
   CCL-LR.py：Only a single LR is used as a JIT defect prediction model  
   CCL-SVM.py：Only a single MLP is used as a JIT defect prediction model  

CCL_imbalanced/:Experimental code of RQ4  
   CCL-weight.py：Weight class unbalance treatment is adopted  
   CCL-ros.py：ROS class unbalance treatment is adopted  
   CCL-rus.py：RUS class unbalance treatment is adopted  
   CCL-SMOTE.py：SMOTE class unbalance treatment is adopted  

CCL_different_SZZ/:Experimental code of RQ5  
   c1-B.py：Raw B-SZZ performance results  
   c1-AG.py：Raw AG-SZZ performance results  
   c1-MA.py：Raw MA-SZZ performance results  
   c1-RA.py：Raw RA-SZZ performance results  
   CCL-B.py：CCL is applied in B-SZZ  
   CCL-AG.py：CCL is applied in AG-SZZ  
   CCL-MA.py：CCL is applied in MA-SZZ  
   CCL-RA.py：CCL is applied in RA-SZZ  
