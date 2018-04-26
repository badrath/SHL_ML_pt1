'''
Created on Jan 13, 2018

python-weka-wrapper3 API documentation:
http://fracpete.github.io/python-weka-wrapper3/api.html

@author: rmarinivision
'''

import weka.core.jvm as jvm
# from weka.classifiers import Classifier
# from weka.datagenerators import DataGenerator
# from weka.core.converters import Loader, Saver
# from weka.clusterers import Clusterer, ClusterEvaluation
from clustering import Clustering

import os
# import sys
# from datetime import datetime

if __name__ == '__main__':
    
    #getting relative paths
    pwd1 = os.getcwd(); #cur dir
    grand_parent_dir = os.path.split(pwd1)[0];
#     out_path = os.path.join(grand_parent_dir,"data","generated_data.arff");   
#     path_DARI_demo_only = os.path.join(grand_parent_dir,"data","DARI_noDate_noString_demo_only.arff");
#     path_demo_an = os.path.join(grand_parent_dir,"data","demo_an_nofid.arff");
    input_filename = "demo_an_nofid.arff";
    path_analysis = os.path.join(grand_parent_dir,"data",input_filename);
    output_dir_path = os.path.join(os.path.split(os.getcwd())[0],"clustered");
    
    #start the jvm, javabridge
    jvm.start();
    
    demo_an_nofid = Clustering(path_analysis, output_dir_path);
    demo_an_nofid.run_SKMeans_137();
    
    #stop jvm, javabridge
    jvm.stop();
    
    print("end of init_practice.py");
    
# #     output_prefix = input_filename.split(".")[0];
# #     output_parent_dir = os.path.join(grand_parent_dir, "clustered");
# #     output_dir_path = os.path.join(output_parent_dir, output_prefix);
#     
# 
#     
#     
#     
#     
# #     #generate data for dev
# #     generator = DataGenerator(classname="weka.datagenerators.classifiers.classification.Agrawal", options=["-B","-P","0.05"]);
# #     DataGenerator.make_data(generator, ["-o",out_path]);
#     
#     #load generated data
#     loader = Loader(classname="weka.core.converters.ArffLoader");
# #     data = loader.load_file(out_path);
# #     data_DARI_demo_only = loader.load_file(path_DARI_demo_only);
#     data_demo_an = loader.load_file(path_analysis); #path_demo_an
#     
#     #cluster using SimpleKMeans (specify 137 clusters)
#     clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans",options=["-N","137"]);
#     clusterer.build_clusterer(data_demo_an); # data
# #     print(clusterer);
#     evaluation = ClusterEvaluation();
#     evaluation.set_model(clusterer);
#     evaluation.test_model(data_demo_an);
#     print("# of clusters: " + str(evaluation.num_clusters));
#     print("log likelihood: " + str(evaluation.log_likelihood));
#     print("cluster assignments:\n" + str(evaluation.cluster_assignments));
# #     print("cluster results:\n " + str(evaluation.cluster_results));
# 
#     #create output dir if it doesn't already exist
#     if(not os.path.exists(output_dir_path)):
#         os.makedirs(output_dir_path);
#         
#     #cluster Instance objects Description of clusters
#     cluster_output_path = os.path.join(output_dir_path, output_prefix + ".skmeans_cl_desc.csv");
#     with open(cluster_output_path, 'w') as outfile:
#         outfile.write(",".join(["cluster_num","distribution\n"])); #header
#         for inst in data_demo_an:    # data
#             cl = clusterer.cluster_instance(inst); # 0-based cluster index
#             dist = clusterer.distribution_for_instance(inst); #cluster membership distribution
#             outfile.write(",".join([str(cl),str(dist)]));
#             outfile.write("\n");
# 
#     #cluster evaluation results
#     cluster_output_path = os.path.join(output_dir_path, output_prefix + ".skmeans_eval.txt");
#     with open(cluster_output_path, 'w') as outfile:
#         outfile.write(",".join(["SKmeans Cluster Evaluation Results\n"])); #header
#         outfile.write(str(evaluation.cluster_results));
# 
#     #cluster assignment by row
#     cluster_output_path = os.path.join(output_dir_path, output_prefix + ".skmeans_cl_assign.csv");
#     with open(cluster_output_path, 'w') as outfile:
#         outfile.write(",".join(["row_num","SKMeans\n"])); #header
#         for i, inst in enumerate(evaluation.cluster_assignments):    # data
#             outfile.write(",".join([str(i),str(inst)]));
#             outfile.write("\n");
# 
#     
#     print("end of init_practice.py");
#     
#     #stop jvm, javabridge
#     jvm.stop();