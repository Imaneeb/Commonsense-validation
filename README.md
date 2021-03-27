# Commonsense-validation

In this project we will be working on commonsense validation and explanation in english, we will train a model to perform different tasks to validate and explain
the commonsense of given sentences. The project consists of two tasks: 
* Task 1: validate whether a sentence makes sense or not.
* Task 2: for the non-sensical sentence choose an option from 3 given option as to why the sentence doesn’t make sense.

## Tasks overview

#### Task 1: Validation
Select the sentence that doens’t make sense out of two statements
* s1: John put a turkey into a fridge.
* s2: John put an elephant into the fridge.

In this example, s1 is a sensical statement, also denoted as sc, while s2 is the nonsensical statement, which is also denoted as sn.
#### Task 2: Explanation multi-option
Select the best reason that explains why the given statement does not make sense.
Nonsensical statement (sn): John put an elephant into the fridge.
* o1: An elephant is much bigger than a fridge.
* o2: Elephants are usually white while fridges are usually white.
* o3: An elephant cannot eat a fridge.

In this example, the option o1 is the correct reason, which is also denoted also as oc, while o2 and o3 are not the reason, which are also denoted as on1 and on2.

## Task1: Commonsense Validation
#### Data understanding:

s1 and s2 are two similar statements that differ by only a few words; one of them makes sense (i.e., conforms to common sense) while the other does not. They are used in our Sub-task A: the Validation sub-task, which requires a model to identify which one makes sense.
Incorrect statement has ’0’ as label and correct statement has ’1’.

#### Model: 

We used RobertaForSequenceClassification which is a RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks. This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

After applying only this model on our data we didn’t get good results by watching the f1-score, For that choose another by adding some layers to the first model and our new model start giving good f1-score each time we augment the number of epochs.

## Task2: Explanation multi-option
#### Data understanding:
For task 2 we’re using the ’SemEval2020-Task2-Commonsense-Validation-andExplanation’ dataset, it contains over 10000 data samples split into train, test and validation sets.
In order to use transformes pretrained models, data should take a specific structure in order to be fed into the model, so a reorganization of our data was necessary, to do that we created a dataset loading script based on the Squad (Stanford Question Answering Dataset) dataset script.

#### Data preprocessing:

The new dataset instance structure is as follows:
- answerKey: a string feature.
- statement: a string feature.
- choices: a dictionary feature containing:
- label: a string feature. default values: ["A", "B", "C"].
- text: a string feature.

The distribution of different sets is:
* Train set: 10000 rows.
* Test set: 1000 rows.
* Validation set: 997 rows.


Tokenization: We used a transformers.Autotokenizer.pretrained() method, it is a generic tokenizer class that will be instantiated as one of the tokenizer classes of
the library when created with the AutoTokenizer.frompretrained() class method.

#### Model:

Model We used a transformers pretrained RobertaForMultipleChoice model, a Roberta Model with a multiple choice classification head on top (a linear layer
on top of the pooled output and a softmax).
