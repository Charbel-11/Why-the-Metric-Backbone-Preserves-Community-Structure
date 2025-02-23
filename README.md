# Why the Metric Backbone Preserves Community Structure
This is the code for the paper [Why the Metric Backbone Preserves Community Structure](https://arxiv.org/abs/2406.03852).

## Experimental Results

To reproduce Figure 1, you can call ```reproduce_results()``` in community_experiments_plots.py. This will generate Figure 1b and c. 
To get the data for Figure 1a, you need to use the MB_Bayesian_Method notebook since graph-tool cannot be used directly with pip. 

## Application to Graph Construction Using q-NN

To reproduce Figures 2, 3 and the numbers of Table 1, you can call ```reproduce_results()``` in SSL.py and TSC.py. 
Note that you need to download the HAR dataset first.

## Computing the Metric Backbone

We provide multiple implementations to compute the metric backbone in both Python and C++ (respectively, in metric_backbone.py and MetricBackboneFast.cpp). This includes a function to get the full metric backbone and one to get the approximate metric backbone.
