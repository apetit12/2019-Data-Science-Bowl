#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is the .py file that generates the submission file

@author: antoinepetit
--
The outcomes in this competition are grouped into 4 groups (labeled 
accuracy_group in the data):

3: the assessment was solved on the first attempt -- num_correct/(num_correct+num_incorrect)=1.0
2: the assessment was solved on the second attempt  -- num_correct/(num_correct+num_incorrect)=0.5
1: the assessment was solved after 3 or more attempts -- num_correct/(num_correct+num_incorrect)<0.5
0: the assessment was never solved -- num_correct+num_incorrect = 0.0

--
For each installation_id represented in the test set, you must predict the 
accuracy_group of the last assessment for that installation_id. The files must 
have a header and should look like the following:

installation_id,accuracy_group
00abaee7,3
01242218,0
etc.
"""

