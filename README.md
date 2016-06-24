## OSVR

Ordinal Support Vector Regression (OSVR) is a general purpose regression model that takes data samples as well as their pairwise ordinal relation as input and output the model parameters learned from data under the max-margin framework. Label (regression response) of each individual sample is not required. OSVR can adapt from no supervision (no labels) to full supervision (labels on all samples). In particular, OSVR can be very useful under weak supervision, where only labels on selected key samples are provided. However, ordinal information should always be available under different supervision settings. 

## How to use

This repository provides an implementation of OSVR in Matlab. The core optimization problem is solved using customized Alternating Direction Method of Multipliers (ADMM).

To see an example, run 'main' script.

## Related publication

If you use the code in your research, please consider citing following publication

Rui Zhao, Quan Gan, Shangfei Wang and Qiang Ji. "Facial Expression Intensity Estimation Using Ordinal Information." IEEE Conference on Computer Vision and Pattern Recognition. 2016. [[PDF](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhao_Facial_Expression_Intensity_CVPR_2016_paper.pdf)]
[[Bib](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Zhao_Facial_Expression_Intensity_CVPR_2016_paper.html)]

## License Conditions

Copyright (C) 2016 Rui Zhao 

Distibution code version 1.0 - 06/25/2016. 
