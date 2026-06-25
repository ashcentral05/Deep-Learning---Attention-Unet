#data reprocessing and load  

#to check if raw image can match with labels
PairCheck.py 

#preprocess
Preprocess.py

#load data
DatasetLoader.py  

#--------------------------------------------------------------------
Pre-processed data can be found via links below:  

train_data_batches.pt https://drive.google.com/file/d/1b0w-of1SJ-ub5mUhuleyV89dPmVgOVXW/view?usp=sharing  
val_data_batches.pt https://drive.google.com/file/d/1tEy6AhQjYc-8gH8hxhLfDWPiNyTSbXqz/view?usp=sharing  
train_label_batches.pt https://drive.google.com/file/d/1X7IFWXm0bgbP7BPMABfMA04e3Jb1KA5Z/view?usp=sharing  
val_label_batches.pt https://drive.google.com/file/d/1XbNXrq_Ji3VV8xDe0YlqSgnsWN3s8fc6/view?usp=sharing


#-----------------------------------------------
If  cuda memory is not enough, you can use the preprocessed data below:
(the image size is reduced to 128*128)

train_data_batches128.pt  
val_data_batches128.pt
train_label_batches128.pt
val_label_batches128.pt
