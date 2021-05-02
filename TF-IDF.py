#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:20:58 2021

@author: meghagupta
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

tfvectorizer = TfidfVectorizer()

msg= ['call you tonight', 'please call a cab', 'please call me.please']
trans = tfvectorizer.fit_transform(msg)

pd.DataFrame(trans.toarray(),columns = tfvectorizer.get_feature_names())
