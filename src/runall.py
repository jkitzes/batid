#!/usr/bin/python

'''
Run all analysis, including fitting classifier and classifying demo data.

File should be run with cwd as src dir.
'''

import os
from classify import write_aml_clean, fit_classifier, classify_calls
from utils import read_params


# ----------------------------------------------------------------------------
# Declare variables
# ----------------------------------------------------------------------------

src_dir = os.path.dirname(__file__)
data_dir = os.path.join(src_dir, os.path.pardir, 'data')
demo_dir = os.path.join(src_dir, os.path.pardir, 'demo')

# Classifier has name, name in aml, and spp to exclude, if any
class_list = [#('sfbay', 'calif', ['MACA','MYCI','EUMA','LAXA','NYFE','NYMA']), 
              ('calif', 'calif', [])]

# ----------------------------------------------------------------------------
# Process reference files and fit classifier 
# ----------------------------------------------------------------------------

n_fits = 100

'''
# Clean aml file for all classifiers. ref-aml-xxxx.txt must be in data dir.
for tclass in class_list:
    aml_path = os.path.join(data_dir, 'ref-aml-' + tclass[1] + '.txt')
    aml_clean_path = os.path.join(data_dir, 'ref-aml-clean-' + tclass[0] + 
                                  '.csv')
    write_aml_clean(aml_path, aml_clean_path, tclass[2])

# Fit each classifier
for tclass in class_list:
    aml_clean_path = os.path.join(data_dir, 'ref-aml-clean-' + tclass[0] + 
                                  '.csv')
    class_path = os.path.join(data_dir, 'class-' + tclass[0] + '.pkl')
    fit_classifier(aml_clean_path, class_path, test=True, performance=True, 
                   n_fits=n_fits, test_split=0.2, save_clf=False)
'''

# ----------------------------------------------------------------------------
# Process demo files with existing config file
# ----------------------------------------------------------------------------

# Get dictionary of parameters (query_dict in GUI)
param_dict = read_params(demo_dir)

# Clean aml file for demo-aml
aml_path = os.path.join(demo_dir, 'demo-aml.txt')
aml_clean_path = os.path.join(demo_dir, 'demo-aml-clean.csv')
write_aml_clean(aml_path, aml_clean_path)

# Classify species in demo
aml_clean_path = os.path.join(demo_dir, 'demo-aml-clean.csv')
class_path = os.path.join(data_dir, 'class-sfbay.pkl')
classify_calls(aml_clean_path, class_path, param_dict['maxqual'])
