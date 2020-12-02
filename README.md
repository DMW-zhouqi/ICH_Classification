# ICH_Classification
Resnet-18 and Densenet-121 under the framework of Pytorch were used for the diagnosis of intracranial hemorrhage and its five subtypes

# Environment requirements
Python >= 3.7 Pytorch >= 1.3 CUDA == 10.0 cuDNN == 7.6

# Files describing
main/test_model_name(e: main_ResNet18) is for training/test files
ROC_model_name(e: ROC_ResNet18) is for Draw ROC curve files
makedatasets and utils files is the auxiliary files, together under a same directory
mydata.zip is a toy_dataset, contains a small amount of data

# Brief operation description
Put all files in the same directory (data files need to be decompressed in advance), run training files, test files and draw ROC files in turn, and you can see the results
