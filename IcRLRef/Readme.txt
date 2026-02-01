This repo contains two datsets. (x_dataset 1) and (x_dataset 2).

Both implement the same architecture and same process. 

For (x_dataset 1), in that folder

Total process: 3 steps. 

1. Process the data: Run Final_submission.ipynb file until cell 10 to get the processed file for further steps.

We have 4 reward function. We have chosen reward function 3 (Dataset 1 reward function 3 folder). Lets see that. Go to reward function 3 folder



2. DDPG network:
   Contains two files : DDPG.py has the architecture, env.py has the enviornmnet. 
   Paste the csv file generated using step 1 in this folder. 
   To trian the ddpg network, in DDPG.py,  Uncomment sess.run(tf.global_variables_initializer()), line 771 and train(), line 773. 
   Once it starts running it will first show you validation acccuracy and then test accuracy. 
   It generates results.csv which includes the results for each participant and gives the prediction of train data "predicted_data_train.csv
   Folder "hard" will contain the trained weights, copy these files and paste into a folder called 'Base_model'for personalisation.   
	

DDPG.py has these functions:  train() - to train (sess.run(tf.global_variables_initializer()) needed to intialise weights)
            Need saver.restore(sess, tf.train.latest_checkpoint(path)) to load the weights for validation /testing
	    eval(i,j)- to validate
            testp(i,j)- test accuracy of each participants
            calculate_se()- Standard error calculation 
            test(i, j)- test accuracy and MRDE of each groups
            eval1(i,j)- this give the folder "predicted_data_train.csv, which is needed for the third step.



DDPG.txt and env.txt explain the architecture. 

3. Personalisation: To personalise first we have to generate our dataset which includes tarin data where MRDE <1 and add test participants data.
   To achive this run personalisation.ipynb which is in the folder reward function 3, it generates files for each participants. For example for Group 1, particpant 17: Group1_17_662_1143_data.csv 
   Now paste Group1_17_662_1143_data.csv into G1_P17 folder. 
	G1_P17 folder has ddpg.py and env.py. 
	Run ddpg.py. It will give you the personalised acccuracy. 

	Here, we dont have to assign any training and test size. It will read automatically from the csv file name. 



Dataset 2: (x_dataset 2)

here we are interested in reward function 4 with visible building width. All personalised files will be found there. 

For datset 2 , "data_processing.ipynb" file will give us the processed file. Dataset 2 has angles in radian.
sec_part_data.csv is the processed file , however we have shuffled the traing set and called it shuffled_sec_part_data.csv , which is used as input for ddpg further . 


Also data_processing.ipynb file also has the personalisation input file generator (cell 10 ), same as before we have to get predicted_data_train.csv (it is in reward function 4 visisble building width ).
 

Similar to datset 1, to personalise, paste the file genarted into respective particiapnts folder and copy the weights from the "hard " folder into base_modle folder in each participant. 
These weights will be used further to personalise. 


