#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:11:18 2021

@author: meghagupta
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

msg= ['call you tonight', 'please call a cab', 'please call me.please']
trans = cv.fit_transform(msg)

pd.DataFrame(trans.toarray(),columns = cv.get_feature_names())
