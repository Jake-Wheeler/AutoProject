The files contained in this repository are varied versions of a project 
intended on analyzing certain text and evaluating its properties in relation 
to its complimentary prompt/question.

NOTE: The code will be buggy without the ongoingly updated compatible versions
for the libraries used. OpenCV works with the language models better around version
3.5-6.

SUMMARY: BlueBox_TextReader

The first successful version that automatically screenshots, seperates, and strips the text 
in two given blue boxes. With a question provided by the user, it analyzes the grammar, 
specificity, relative sensibility, and other aspects of the text and its relation to the 
question. Among the statistics provided by the code, it also determines which response was 
better and pressents its choice as well as the calculated difference between the two from 
their overall quality. It should be noted that an effort was made to provide a suggested 
improvement upon the text by the code, but unfortunately not enough time has been spent to 
get it to work yet.

SUMMARY: prompt-response_eval.py/ipynb

Both the python file as well as its jupyter notebook version are the
same. This project version removed the aspect in the first version with taking a screenshot
and seperating the text. It simply relies on a prompt and two responses to be entered by the
user. Evaluating and presenting the results in the same manner as BlueBox_TextReader.
