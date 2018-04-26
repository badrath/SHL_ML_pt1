'''
Created on Apr 25, 2018

A class to hold all clustering code and contain the data, input and output.

@author: rmarinivision
'''

from weka.clusterers import Clusterer, ClusterEvaluation
from weka.core.converters import Loader
import copy
import os
from datetime import datetime

class Clustering(object):
    '''
    classdocs
    '''

    def __init__(self, input_path, output_dir):
        '''
        Constructor
        '''
        self.input_path = input_path;
        loader = Loader(classname="weka.core.converters.ArffLoader");
        self.data_loaded = loader.load_file(self.input_path);
        self.output_dir = output_dir;
        
    def run_SKMeans_137(self):
        
        #construct output paths
        output_prefix = os.path.split(self.input_path)[-1].split(".")[0];
        print(output_prefix);
        write_date = output_prefix + "." + str(datetime.now().date());
        SKMeans_dir = os.path.join(self.output_dir,"SKMeans");
        eval_path = os.path.join(SKMeans_dir, write_date + ".cl_eval.txt");
        clust_desc_path = os.path.join(SKMeans_dir, write_date + ".cl_descr.txt");
        clust_assign_path = os.path.join(SKMeans_dir, write_date + ".cl_assign.txt");
        
        #create output dir if it doesn't already exist
        if(not os.path.exists(SKMeans_dir)):
            os.makedirs(SKMeans_dir);
        
        #clone data and build clusters
#         data_clone = copy.deepcopy(self.data_loaded);
        data_clone = self.data_loaded;
        clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N","137"]);
        clusterer.build_clusterer(data_clone);
        
        #cluster evaluation
        evaluation = ClusterEvaluation();
        evaluation.set_model(clusterer);
        evaluation.test_model(data_clone);
        with open(eval_path, 'w') as outfile:
            outfile.write("number of clusters: \t" + str(evaluation.num_clusters) + "\n");
            outfile.write("log likelihood: \t" + str(evaluation.num_clusters) + "\n");
            outfile.write("cluster assignments: \t" + str(evaluation.cluster_assignments) + "\n");
            outfile.write("***********************\n")
            outfile.write("\t".join(["SKmeans Cluster Evaluation Results\n"])); #header
            outfile.write(str(evaluation.cluster_results) + "\n");
        
        #cluster Instance objects Description of clusters
        with open(clust_desc_path, 'w') as outfile:
            outfile.write(",".join(["cluster_num","distribution\n"])); #header
            for inst in data_clone:    # data
                cl = clusterer.cluster_instance(inst); # 0-based cluster index
                dist = clusterer.distribution_for_instance(inst); #cluster membership distribution
                outfile.write(",".join([str(cl),str(dist)]));
                outfile.write("\n");
     
        #cluster assignment by row
        with open(clust_assign_path, 'w') as outfile:
            outfile.write(",".join(["row_num","SKMeans\n"])); #header
            for i, inst in enumerate(evaluation.cluster_assignments):    # data
                outfile.write(",".join([str(i),str(inst)]));
                outfile.write("\n");
        
        
        return();
        