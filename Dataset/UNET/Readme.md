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

train_data_batches128.pt https://drive.google.com/file/d/1RCi5UeItYTlwhJsC88GwB8STcTVWUyDC/view?usp=sharing  
val_data_batches128.pt https://drive.google.com/file/d/1QJLd6jxaAivr7B8nr-E-mJVZ1nYQGoKK/view?usp=sharing
train_label_batches128.pt https://drive.google.com/file/d/1yiGHi_mEfT3ZUUYm6O8SCRLEMWp_wjbp/view?usp=sharing
val_label_batches128.pt https://drive.google.com/file/d/1ZspbELQuYvqz82YG7MdXuUNdzZ_sjH7O/view?usp=sharing
