

Introduction (4 pts):


(only done cnn)Model Figure (4 pts):

  Use Alexnet to embed the pictures and use one-hot embedding to embed the captions, followed by a positional encoding layer, then put the source and target to a transformer layer. Finally, use a fully connected layer to get the predicted words (in 51 classes).












(only done cnn)Model Parameters (4 pts): 

Character by character: (128*(51+1) + (1+1)*256*128 + (128 *(sequence_len+1)*3 + 128 * (3*3+1) *3 + 2* 128*(3*3+1)  + 128*(sequence_len+1)) *6*8  + 128*(sequence_len+1) *128 + 128*(sequence_len+1) * 51) * N

Note: all (+1) are for biases
(128*(51+1): caption embedding (+1 bias, fully connected layer)
(1+1)*256*128: image embedding(CNN)
(128 *(sequence_len+1)*3 self attention for target(embedded caption)
128 * (3*3+1) *3 + self attention for source(embedded image)
2* 128*(3*3+1)  + 128*(sequence_len+1) attention layer for combining the output of encoder and the decoder
128*(sequence_len+1) *128 fully connected layer 2
128*(sequence_len+1) * 51 fully connected layer 3 to get the output probability


Model Examples (4 pts): 

Examples of how the model performs on two actual examples from the test set: 
One successful
One unsuccessful

Data Source (1 pt):

We collect data from a github
(https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)

Data Summary (4 pts):

Our dataset has 8,091 pictures that are all creatures.
The data have two parts, one of which is images of various dimensions, the other is a text file with captions corresponding to every image.
We notice that similar creatures are clustered, for example the first 100 pictures are mostly dogs and the last 100 pictures have no dogs.

Data Transformation (3 pts):

For source we crop the square area in the center and resize it to 224x224.
For target we create a vocabulary for every character and turn the caption into a one-hot embedding for each character

Data Split (2 pts):
	 Due to clustering we talked about in data summary, we shuffle all of the 8071 data, then split as follows
Train: first 6500
Valid: 6500th-7500th
Test: last 591(all that remains)


Training Curve (4 pts):

A curve that shows both training and validation performance

Hyperparameter Tuning (4 pts):



(Only done CNN) Quantitative Measures (2 pts):

A description and justification of the quantitative measure that you are using to evaluate your results



(Only done CNN) Quantitative and Qualitative Results (8 pts):
Describe the quantitative and qualitative results. You may choose to use a table or figure to aid in your description


Justification of Results (20 pts):

  A justification that your implemented method performed reasonably, given the difficulty of the problem—or a hypothesis for why it doesn’t

  It performs reasonably on smaller size data(<100),but not very well in larger datasets, for the full dataset we talked about above, the model performs poorly when given only the start token, but when providing the target caption (with mask), it produces reasonable results.

 Note: We use ‘{‘ for start character and ‘}’ for end character, and ‘@’ for padding. 
The first 3 rows is the sentence generated by our model providing only the start token,
The 4th row is the sentence generated providing the masked target caption.
The final row is the target caption.
We suspect that this is because the error produced early in a sentence will lead to further mistakes later in the sentence.
However, this results seems to be addressable by smaller sample size, so we think that the hyperparameter should be tuned by bigger dataset, but 50 epochs takes 90 minutes already, we have tried a few hyperparameters. For example, using learning rate of 0.001 for 500 epochs, it takes the whole night to train, but the loss curve still fluctuates heavily, which implies we should use a lower learning rate, 
Ethical Consideration (4 pts):

Description of a use of the system that could give rise to ethical issues


Authors (2 pts):

A description of how the work was split



ReadMe

Image Caption by using CNN & Transformer 


Model Structure： CNN Encoder Model & RNN Decoder Model


By using our constructed models, analyzing the content contained in the images.


For a computer, it does not have as much experience with things as humans do. So it is not a simple task when you ask a computer to analyze a picture and tell all the contents in the picture. It is like a human baby, it doesn't know what's in the picture until it does the learning.


Purpose:
  Therefore, we have developed a machine learning system that can learn unsupervised and thus do the job of representing the content of a picture as a text. It can do sorting and reading. It also has great significance for society. By converting pictures to text and then text to speech, it can produce the information stored in the pictures easily accessible to people with visual disabilities.

 We are aiming to train a transformer model that generates reasonable captions outputs from picture inputs, while investigating the impacts of grouping image dataset on accuracy of captions.
  The model will not be limited by the amount of elements within a picture, where the images contain creatures, and images that only contain items instead of creatures, the goal for the model is to distinguish the difference.

Model tasks:
  The task aims to train a RNN model that receives a picture as input then generates a reasonable caption for that picture. Also to investigate whether grouping the image first helps with the quality of caption.
  Since we want to challenge the capacity of the model, we want to build a model that is not limited to the elements in the pictures. Therefore, we choose images that contain creatures, and images that only contain items instead of creatures. We hope the model we build can tell the difference between different images accurately.


Model explained:
  We will train two different models based on two different dataset, one is for creatures, the other is for items. We use CNN model to justify whether this image belongs to the category of creatures or items. In this model, we use the basic model without back normalization or dropout. Each model will contain a couple of CNN layers followed by an attention mechanism to embed our image, followed by an RNN layer(transformer). Then we will train another model to classify whether the input image is a creature or an item, then put it to the correct model described above to generate the final caption.

  We will also combine the two dataset to form a single huge dataset and try to train a single model based on this large dataset to compare the results with the separate one.


Dataset:
  There are two packages of images. The pictures in one of the datasets are all about creatures. The other one is about items.
Firstly, all images are stored together. Then they are shuffled since they should be randomly stored in different groups, such as training set, validation set and testing set.. Therefore, we can use an exact dataset to train the model or test the model. It can help us build a better model which is more efficient.
Not only the relevant photos are contained in each dataset, but also containing a file with the corresponding caption for each photo.

  The first dataset contains a folder with 8090 photographs in JPEG format and a text file with their corresponding captions. All these photographs contain at least a person or an animal. The second dataset contains 11577 photographs in JPEG format. All these photographs contain merchandise like carpets, stamps, and jewelry. Therefore, the total number of images is 19667. After shuffling, the training set has 13766 images, validation set has 3933 images, testing set has 1968 images.

All data is collected from these two sources. (https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names)
Todo ( another one source from, 数据items那个网站):


CNN Model


Pal, L. (2021, December 11). Image classification: A comparison of DNN, CNN and Transfer Learning approach. Medium. https://medium.com/analytics-vidhya/image-classification-a-comparison-of-dnn-cnn-and-transfer-learning-approach-704535beca25


Here is a simple description of CNN. An image is divided into a few parts. By analyzing those parts, the model can predict what this figure is.


Model Parameters:
Model: what model we used to do training.
Train_data: dataset that we used for training data.
Val_data: dataset that we used for validation data.
Batch_size=1024:  defines the number of samples that will be propagated through the network. 
Weight_decay=0.0:  penalize complexity        
Optimizer="sgd": used to change the attributes of the neural network such as weights and learning rate to reduce the losses (Stochastic Gradient Descent)
Learning_rate=0.1: a configurable hyperparameter used in the training of neural networks that has a small positive value
Momentum=0.3: a variant of the stochastic gradient descent
Data_shuffle=True: doing shuffle
Num_epochs=3:  a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset
Checkpoint_path=None: saving data, can be used quickly later.



Figures:






Data Summary:
By the third figure, we can know that the training accuracy and validation accuracy are 94% and 94%, separately. In 19667 images, only 1180 images are predicted incorrectly. It is a great result. Because we have high accuracy while it is not overfit. The loss we get is around 0.0007 which is low enough. Therefore, the model is very effective..

The size of images we download are not 224 * 224. Therefore, we use a tool to adjust the size of them. In this process, there are two images that can not be adjusted. We decide to remove them in order to all 224 * 224 images. Then we use Imagefolder and Dataloader to transformate the image so that we can use tensor to do the tasks.

We were using 0.0001 as our learning rate, but we found out that it took too much time on training. So we decided to use a larger learning rate which is 0.001. It can reduce the time, but bring  the cost of arriving at the final set of weights. We were using 256 as our batch size. It produced too many iterations, though the accuracy didn’t change after the 80th iteration. So we decided to use 1024 as our batch size which brought less iteration and saved some time.


Successful example:
Here is one figure that we successfully predicted. 
<img width="209" alt="ph2" src="https://user-images.githubusercontent.com/71194632/163875823-c161b027-911d-4619-b489-15847a10713f.png">


Unsuccessful example:
Here is one figure that we unsuccessfully predicted. There are hands and clothes in this picture. Therefore, the model doesn’t know whether it should be determined as creatures or items.
<img width="204" alt="ph1" src="https://user-images.githubusercontent.com/71194632/163875691-e8c77962-bb21-46ef-9f56-982666be3e55.png">


Quantitative measures
The model is able to process approximately 20000 images in a decent amount of time, with high accuracies, which means the differentiation with CNN model and captions made by RNN model is highly accurate. With the CNN model, we have tried to predict with mini-batch first, then increase the batch size as we tune the parameters and optimize the model, which now can process all images within the image dataset. 

Quantitative and qualitative results
Quantitative results: 
As mentioned in the above question, the model is able to process 19667 images within a dataset with high accuracies:
CNN model training accuracy: 95%
CNN model validation accuracy: 95%
The model accuracy for mini-batch at approximately 200 images is 93%, after increasing the batch size to 19667, the accuracy was still high, which demonstrated the stability of this model.
The running time with MLP at 76 iterations with MacBook M1 Chip is 23.4 seconds after several optimizing and parameter tuning.

Qualitative results:

This is an unsuccessful example of the CNN model, tensor([0]) represents creatures, and tensor([1]) represents items. From this example we can tell, by human eyes, a lady doing her makeup at a bar, which is for sure to be classified into the creature category. For some reason, maybe due to colour elements, the model classified this image as an item, this is due to high but not 100% accuracy. 




RNN Model

Figures:

Data Summary

Successful example:

Unsuccessful example:






Ethical implications:
  First of all, image caption is a function that we use in people's usual lives. Although machine learning nowadays is powerful enough to do captions for many images, it is hard for a model to take parameters that it never learned about or something abstract. Therefore, we certainly do not recommend anyone who purely and fully relies on this model for commercial use. This is a disclaimer to all people that we are not responsible for any kind of personal business loss over this machine learning model.

  People may use this model to generate useless comments on social media like Instagrams and Twitter, this will make the poster confused and waste their time on replying to this. 




Authors:
Yulin Wang: 

Tian Ze Jia: Building CNN model, training the model and tuning the hyperparameters. Tested the model and produced the successful and unsuccessful examples. Write the final report.

Hongwei Wen: 

Shijia Wu: Building CNN model to justify difference between the category of images. Searching, cleaning, resizing data as the size we need. Testing CNN model and fixing errors. Write the final report. 


