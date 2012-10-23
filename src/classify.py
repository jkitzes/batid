#!/usr/bin/python

'''
Use saved random forest model to predict identities of data_features calls.
'''

from __future__ import division
import os
import numpy as np
from matplotlib.mlab import csv2rec, rec2csv, rec_drop_fields
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from utils import read_params, sum_group


# ----------------------------------------------------------------------------
# Functions 
# ----------------------------------------------------------------------------

def main():
    '''Perform species classification.'''

    # Get dictionary of paramters (query_dict in GUI)
    cwd = os.getcwd()
    param_dict = read_params(cwd)

    # Get paths
    output_dir, aml_path = os.path.split(param_dict['filepath'])
    aml_clean_path = aml_path.split('.')[0] + '-clean.csv'
    
    file_dir = os.path.dirname(__file__)
    class_dir = os.path.join(file_dir, os.path.pardir, 'data')
    class_path = os.path.join(class_dir, param_dict['classifier'])

    # Write scan clean file
    write_aml_clean(aml_path, aml_clean_path)

    # Run classifier and generate output
    classify_calls(aml_clean_path, class_path, param_dict['maxqual'])
    

def write_aml_clean(aml_path, aml_clean_path, folder_exclude=[]):
    '''
    Convert AML txt file to cleaned csv.
    
    Cols are same, except cleaned file has 2 cols at front for path and folder 
    associated with each file.

    Note that when making cleaned reference file, the folder col will contain 
    the species code ASSUMING THAT the reference calls were organized into 
    folders, each with a species code.
    '''

    # Open input and output files
    file_in = open(aml_path, 'rU')
    file_out = open(aml_clean_path, 'w')

    # Write header in file_out
    header = ('Path,Folder,Filename,st,Dur,TBC,Fmax,Fmin,Fmean,Tk,Fk,Qk,Tc,' 
              'Fc,Dc,S1,Sc,Qual,Pmc')
    file_out.write(header + '\n')

    # Read each line in file_in and write to file_out when appropriate
    for line in file_in:

        # If line starts with Ana, space, or newline, skip (not a call)
        if line[0:3] == 'Ana' or line[0] == ' ' or line[0] == '\n':
            continue

        # If line starts with ! (new folder), extract some info and skip
        if line[0] == '!':
            path = line[1:-1]  # Everything except ! and \n
            folder = line.split('\\')[-2:-1][0]  # Last item is \n
            continue

        # If folder in folder_exclude, continue (used to exclude spp in ref)
        if folder in folder_exclude:
            continue

        # All other lines are call, add 2 leading cols and write
        line_list = line.replace(' ','').split('\t')[:-1]  # Drop \n
        line_com = ''.join([x+',' for x in line_list])[:-1] # Drop last ,

        final_line = (path + ',' + folder + ',' + line_com)
        file_out.write(final_line + '\n')

    file_in.close()
    file_out.close()


def fit_classifier(aml_clean_path, class_path, test=False, performance=False, 
                   n_fits=100, test_split=0.2, save_clf=True):
    '''Fits random forest classifier to aml_ref_clean formatted csv.
    Note that the species code should be contained in the folder col.'''

    # Get class_path dir, used for ancillary file names
    class_dir, tail = os.path.split(class_path)
    prefix = tail.split('.')[0]

    # Load refe_features_table
    table = csv2rec(aml_clean_path)

    # Only use calls with qual < 0.3 (Armitage)
    table = table[table.qual < 0.3]

    # Get target col (y) with integer codes instead of spp names
    y_str = table.folder  # Assumes spp name is in folder col
    y_str_uniq = set(list(y_str))

    n_spp = len(y_str_uniq)
    spp_codes = range(0, n_spp)
    code_table = np.array(zip(spp_codes, y_str_uniq),
                          dtype = [('code','<i8'), ('spp', '|S8')])

    y = np.zeros(len(y_str))  # Get col of full length with codes, not names
    for code, spp in code_table:
        y[y_str == spp] = int(code)

    # Get filename col for later grouping into passes
    f = table.filename

    # Remove non-feature cols from table
    table = rec_drop_fields(table, ['path', 'folder', 'filename', 'st', 'dc', 
                                    'qual', 'pmc'])

    # Get list of feature names remaining in table
    feature_names = table.dtype.names

    # Recarray to ndarray - http://stackoverflow.com/questions/5957380/
    # convert-structured-array-to-regular-numpy-array
    X = table.view((float, len(table.dtype.names)))

    # Partition data if test, holding portion for testing
    if not test:
        X_tr = X
        y_tr = y
        f_tr = f
        X_te = X
        y_te = y
        f_te = f
    else:
        # Use StratifiedShuffleSplit since train_test_split does not stratify
        sss = StratifiedShuffleSplit(y, 1, test_size=test_split)
        for train_index, test_index in sss:  # Only once since n_iter=1 above
            X_tr, X_te = X[train_index], X[test_index]
            y_tr, y_te = y[train_index], y[test_index]
            f_tr, f_te = f[train_index], f[test_index]

        sort_ind = f_te.argsort()  # Sort test data for pass analysis later
        X_te = X_te[sort_ind,:]  # Sort rows
        y_te = y_te[sort_ind]
        f_te = f_te[sort_ind]
        # (Train data order does not matter)

    # Define and fit classifier
    clf = RandomForestClassifier(n_estimators=n_fits, oob_score=True, 
                                 compute_importances=True)
    clf.fit(X_tr, y_tr)

    # If performance, save various performance metrics
    # NOTE: Performance of passes is difficult to understand if if test=True,
    # as the calls in one pass may be split up.
    if performance:

        # Get OOB score
        print 'OOB Score: ', clf.oob_score_

        # Predict on test data, which may be held out (test=True) or all data
        y_te_pr = clf.predict(X_te)

        # Get true data and predictions by passes
        pred_te = clf.predict_proba(X_te)  # Prob of each spp
        f_te_p, pred_te_p, other = sum_group(f_te, pred_te, [y_te])
        y_te_p = other[0]  # Actual spp for each pass

        y_te_p_pr = []
        for row in xrange(len(y_te_p)):  # Find pred species for each pass
            y_te_p_pr.append(pred_te_p[row].argmax())  # First ind, ties bias
        y_te_p_pr = np.array(y_te_p_pr)

        # Get accuracy and confusion matrix for calls
        def make_conf_mat(y_te, y_te_pr, type):
            conf_mat = metrics.confusion_matrix(y_te, y_te_pr)
            conf_mat_frac = conf_mat / np.sum(conf_mat, axis=0)
            print type, ' Accuracy: ', metrics.zero_one_score(y_te, y_te_pr)

            np.savetxt(os.path.join(class_dir, prefix+'_conf_'+type+'.csv'),
                       conf_mat, fmt='%i', delimiter=',')
            np.savetxt(os.path.join(class_dir, prefix+'_conffr_'+type+'.csv'), 
                       conf_mat_frac, fmt = '%.6f', delimiter=',')

        make_conf_mat(y_te, y_te_pr, 'call')
        make_conf_mat(y_te_p, y_te_p_pr, 'pass')

    # Save spp_code table, feature_names, and pickle classifier
    rec2csv(code_table, os.path.join(class_dir, prefix + '_spp_codes.csv'))
    rec2csv(np.array(list(feature_names), dtype=[('features', 'S8')]),
        os.path.join(class_dir, prefix + '_feature_names.csv'))
    if save_clf:
        joblib.dump(clf, class_path, compress = 9)


def classify_calls(aml_clean_path, class_path, maxqual=0.3):
    '''Classify calls by species and save files.'''

    # Get dirs
    output_dir, tail = os.path.split(aml_clean_path)
    class_dir, tail = os.path.split(class_path)
    prefix = tail.split('.')[0]

    # Load spp_names as list
    spp_path = os.path.join(class_dir, prefix + '_spp_codes.csv')
    spp_names = list(csv2rec(spp_path).spp)
    spp_names_comma = ''.join([x + ',' for x in spp_names])[:-1]

    # Load classifier from pickle
    clf = joblib.load(class_path)

    # Load aml_clean as recarray
    table = csv2rec(aml_clean_path)

    # Only use calls with qual < maxqual
    table = table[table.qual < float(maxqual)]

    # Save path, folder, call, and qual fields for later
    path = table.path
    folder = table.folder
    call = table.filename
    qual = table.qual

    # Remove non-feature cols from table
    table = rec_drop_fields(table, ['path', 'folder', 'filename', 'st', 'dc', 
                                    'qual', 'pmc'])

    # Recarray to ndarray, since classifier required ndarray
    X = table.view((float, len(table.dtype.names)))

    # Predict probabilities for each call
    pred = clf.predict_proba(X)

    # Save call_prob and call_bin files
    header = 'path,folder,pass,qual,' + spp_names_comma

    file_callpr = open(os.path.join(output_dir, 'call_prob.csv'), 'w')
    file_callbi = open(os.path.join(output_dir, 'call_bin.csv'), 'w')

    file_callpr.write(header + '\n')
    file_callbi.write(header + '\n')

    for row in xrange(0, len(call)):  # For all calls
        row_comma_prob = ''.join([str(x)+',' for x in pred[row]])[:-1]
        row_bin = (pred[row] == pred[row].max()) + 0  # +0 makes int not bool
        row_comma_bin = ''.join([str(x)+',' for x in row_bin])[:-1]

        file_callpr.write(path[row] + ',' + folder[row] + ',' + call[row] + 
                          ',' + str(qual[row]) + ',' + row_comma_prob + '\n')
        file_callbi.write(path[row] + ',' + folder[row] + ',' + call[row] + 
                          ',' + str(qual[row]) + ',' + row_comma_bin + '\n')

    file_callpr.close()
    file_callbi.close()

    # Summarize calls into passes
    other_cols = [path, folder]
    callp, predp, otherp = sum_group(call, pred, other_cols)
    callp2, qualp = sum_group(call, qual)[0:2]

    # Save pass_prob and pass_bin file
    pathp = otherp[0]
    folderp = otherp[1]

    header = 'path,folder,pass,ncalls,qual,' + spp_names_comma

    file_passpr = open(os.path.join(output_dir, 'pass_prob.csv'), 'w')
    file_passbi = open(os.path.join(output_dir, 'pass_bin.csv'), 'w')

    file_passpr.write(header + '\n')
    file_passbi.write(header + '\n')

    for row in xrange(0, len(callp)):  # For all calls
        ncalls = np.sum(predp[row])
        tpredp = predp[row] / ncalls  # Mean predp
        tqual = qualp[row] / ncalls  # Mean qual

        row_comma_prob = ''.join([str(x)+',' for x in tpredp])[:-1]
        row_bin = (predp[row] == predp[row].max()) + 0  # +0 makes int not bool
        row_comma_bin = ''.join([str(x)+',' for x in row_bin])[:-1]

        file_passpr.write(pathp[row] + ',' + folderp[row] + ',' + callp[row] + 
                          ',' + str(ncalls) + ',' + str(tqual) + ',' + 
                          row_comma_prob + '\n')
        file_passbi.write(pathp[row] + ',' + folderp[row] + ',' + callp[row] + 
                          ',' + str(ncalls) + ',' + str(tqual) + ',' + 
                          row_comma_bin + '\n')

    file_passpr.close()
    file_passbi.close()


# ----------------------------------------------------------------------------
# Run main (if file run as script)
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
