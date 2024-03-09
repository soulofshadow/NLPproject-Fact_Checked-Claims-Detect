import pdb
import logging
import argparse
import os
import pandas as pd
import numpy as np
from trectools import TrecRun, TrecQrel, TrecEval
from os.path import join, dirname, abspath

import sys
sys.path.append('.')
from format_checker.main import check_format
from scorer.utils import print_thresholded_metric, print_single_metric
"""
Scoring of Task 2 with the metrics Average Precision, R-Precision, P@N, RR@N. 
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50, 1000]

def evaluate(gold_fpath, pred_fpath, thresholds=None):
    """
    Evaluates the predicted line rankings w.r.t. a gold file.
    Metrics are: Average Precision, R-Pr, Reciprocal Rank, Precision@N
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number at each line, where the list is ordered by check-worthiness.
    :param thresholds: thresholds used for Reciprocal Rank@N and Precision@N.
    If not specified - 1, 3, 5, 10, 20, 50, len(ranked_lines).
    """
    gold_labels = TrecQrel(gold_fpath)
    prediction = TrecRun(pred_fpath)
    results = TrecEval(prediction, gold_labels)

    # Calculate Metrics
    maps = [results.get_map(depth=i) for i in MAIN_THRESHOLDS]
    mrr = results.get_reciprocal_rank()
    #precisions = [results.get_precision(depth=i) for i in MAIN_THRESHOLDS]

    return maps, mrr, 1

def evaluate2(gold_fpath, pred_fpath, thresholds=None, trec_eval=True):
    """
    Evaluates the predicted line rankings w.r.t. a gold file.
    Metrics are: Average Precision, R-Pr, Reciprocal Rank, Precision@N
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number at each line, where the list is ordered by check-worthiness.
    :param thresholds: thresholds used for Reciprocal Rank@N and Precision@N.
    If not specified - 1, 3, 5, 10, 20, 50, len(ranked_lines).
    """
    gold_labels = TrecQrel(gold_fpath)
    print(gold_labels.qrels_data.head())
    prediction = TrecRun(pred_fpath)
    #print(prediction.run_data)
    results = TrecEval(prediction, gold_labels)


    dept = 5
    label = "MAP@%d" % (dept)

    # We only care for binary evaluation here:
    relevant_docs = gold_labels.qrels_data[gold_labels.qrels_data.rel > 0].copy()
    relevant_docs["rel"] = 1

    if trec_eval:
        trecformat = prediction.run_data.sort_values(["query", "score", "docid"], ascending=[True,False,False]).reset_index()
        topX = trecformat.groupby("query")[["query","docid","score"]].head(dept)
    else:
        topX = prediction.run_data.groupby("query")[["query","docid","score"]].head(dept)

    # check number of queries
    print("Prediction topics: ",prediction.topics())
    nqueries = len(prediction.topics())

    # Make sure that rank position starts by 1
    topX["rank"] = 1
    topX["rank"] = topX.groupby("query")["rank"].cumsum()
    topX["discount"] = 1. / np.log2(topX["rank"]+1)
    
    print("Topx:", topX)

    # Keep only documents that are relevant (rel > 0)
    selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left")

    selection["rel"] = selection.groupby("query")["rel"].cumsum()
    # contribution of each relevant document
    selection[label] = selection["rel"] / selection["rank"]

    print("Selection ",selection)
    # MAP is the sum of individual's contribution
    map_per_query = selection[["query", label]].groupby("query").sum()
    print("map per query: ",map_per_query)
    relevant_docs[label] = relevant_docs["rel"]
    nrel_per_query = relevant_docs[["query",label]].groupby("query").sum()
    
    print(map_per_query.sum())

    map_per_query = map_per_query / nrel_per_query
    print(map_per_query.sum())
    print(nqueries)
    # Calculate Metrics
    #maps = [results.get_map(depth=i) for i in MAIN_THRESHOLDS]
    #mrr = results.get_reciprocal_rank()
    #precisions = [results.get_precision(depth=i) for i in MAIN_THRESHOLDS]

    #return maps, mrr, precisions
    return 1

def validate_files(pred_file, gold_file):
    if not check_format(pred_file):
        logging.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        return False

    # Checking that all the input tweets are in the prediciton file and have predicitons. 
    pred_names = ['iclaim_id', 'zero', 'vclaim_id', 'rank', 'score', 'tag']
    pred_df = pd.read_csv(pred_file, sep='\t', names=pred_names, index_col=False)
    gold_names = ['iclaim_id', 'zero', 'vclaim_id', 'relevance']
    gold_df = pd.read_csv(gold_file, sep='\t', names=gold_names, index_col=False)
    for iclaim in set(gold_df.iclaim_id):
        if iclaim not in pred_df.iclaim_id.tolist():
            logging.error('Missing iclaim {}. Cannot score.'.format(iclaim))
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="Path to files with gold annotations.")
    
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="Path to files with ranked line_numbers.")

    args = parser.parse_args()

    line_separator = '=' * 120
    pred_file = args.pred_file_path
    gold_file = args.gold_file_path

    if validate_files(pred_file, gold_file):
        maps, mrr, precisions = evaluate(gold_file, pred_file)
        filename = os.path.basename(pred_file)
        logging.info('{:=^120}'.format(' RESULTS for {} '.format(filename)))
        print_single_metric('RECIPROCAL RANK:', mrr)
        print_thresholded_metric('PRECISION@N:', MAIN_THRESHOLDS, precisions)
        print_thresholded_metric('MAP@N:', MAIN_THRESHOLDS, maps)

