The NERTestDataLabels contains the ground truth labels (Entity labels) for each word and is used for evaluating the performance of SANS for individuals words. 

The NERTestDataText contains the text (i.e., news reports) that is used to evaluate the SANS. 

Two files are created so that it is easier to compare the performance of SANs and other off-the shelf NER as the off-the shelf NER systems require  whole sentence/news reports as inputs in order to predict the entity labels for each word rather than individual entries as in NERTestDataLabels. 