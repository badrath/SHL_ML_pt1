'''
Created on Jan 13, 2018

python-weka-wrapper3 API documentation:
http://fracpete.github.io/python-weka-wrapper3/api.html

@author: rmarinivision
'''
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.datagenerators import DataGenerator
from weka.core.converters import Loader, Saver
from weka.clusterers import Clusterer

import os
import sys

if __name__ == '__main__':
    
    #getting relative paths
    pwd1 = os.getcwd(); #cur dir
    grand_parent_dir = os.path.split(pwd1)[0];
    out_path = os.path.join(grand_parent_dir,"data","generated_data.arff");    
    
    #start the jvm, javabridge
    jvm.start();
    
    
    #generate data for dev
    generator = DataGenerator(classname="weka.datagenerators.classifiers.classification.Agrawal", options=["-B","-P","0.05"]);
    DataGenerator.make_data(generator, ["-o",out_path]);
    
    #load generated data
    loader = Loader(classname="weka.core.converters.ArffLoader");
    data = loader.load_file(out_path);
    
    #cluster using SimpleKMeans (specify 3 clusters)
    clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans",options=["-N","3"]);
    clusterer.build_clusterer(data);
    print(clusterer);
    
    #cluster Instance objects
    for inst in data:
        cl = clusterer.cluster_instance(inst); # 0-based cluster index
        dist = clusterer.distribution_for_instance(inst); #cluster membership distribution
        print("cluster = " + str(cl) + ", distribution = " + str(dist));

    
    print("end of init_practice.py");
    
    #stop jvm, javabridge
    jvm.stop();