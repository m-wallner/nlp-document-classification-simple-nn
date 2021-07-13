# Natural Language Processing: Document Classification Using a Simple Neural Network

The goal of this project was to create a complete NLP pipeline for document classification using the pre-trained vectors of a word embedding model.

## Preprocessing
First, the data set is loaded, all documents get tokenized, and a dictionary of vocabularies is created from the tokenized text, with tokens with low frequencies being excluded. Next, a lookup for the embeddings of all the words in the dictionary is created – this is an embedding matrix that maps the ID of each word to the respective pre-trained vector from the embedding model, which is GloVe with a vector length of 300 in this case. Words that are not found in the embedding model are replaced by randomly initialized vectors. The preprocessed and embedded data is then saved to save time in future runs, and a PyTorch Dataset object is created for the training, validation and test set for optimized data loading during training and inference time.

## Training the model
The average over all separate word vectors in a document is calculated and used as the vector representation of this document during training and inference time, using PyTorch’s EmbeddingBag class. For document classification, a simple linear layer is used that maps the document embedding to the total of 12 available classes. As criterion, Cross Entropy Loss is then used instead of Negative Log Likelihood Loss for the final document classification task, to save an additional layer in the neural network. This means that the class with the highest probability is chosen as the final model output. Furthermore, Adam is used as optimization mechanism.
After each epoch, the model’s accuracy is evaluated on the validation set. If the current model’s performance is better than that of all previous ones, it is saved. Furthermore, early stopping is implemented in case the model does not improve for 5 consecutive epochs.
