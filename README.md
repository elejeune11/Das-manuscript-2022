# Introduction
This repository contains code used to analyze mechanical structure and mechanical function data in the manuscript "Mechanical response of cardiac microtissues to acute localized injury" by Shoshana L Das, Bryan Sutherland, Emma Lejeune, Jeroen Eyckmans, and Christopher S Chen (link forthcoming). There are two separate codes included in this repository -- see the "Object_Segmentation" folder and the "Strain_Analysis" folder. 

# Part 1: Object Segmentation
The Jupyter notebook `approximate_myofibril_segmentation.ipynb` contains code to segment and quantify "myofibril-like" structures. Details are contained within the Jupyter notebook, and the process is explained in text in the manuscript. For context, the goal of this code is to determine the total length of segmented fiber-like structures that are over 6 microns long. 

<p align="center">
<img src="https://github.com/elejeune11/Das-manuscript-2022/blob/main/Object_Segmentation/segmentation_example.png" width="500"/>
</p>

# Part 2: Strain Analysis
The Jupyter notebook `compute_regional_strain.ipynb` contains code to compute strain within specified regions of interest in the microtissue domain. The Jupyter notebook also uses functions specified in `compute_strain_additional_functions.py'. 

ROI definition: 
<p align="center">
<img src="https://github.com/elejeune11/Das-manuscript-2022/blob/main/Strain_Analysis/D1T3_Before_regions.png" width="800"/>
</p>

Strain in each ROI: 
<p align="center">
<img src="https://github.com/elejeune11/Das-manuscript-2022/blob/main/Strain_Analysis/D1T3_Before_strain.png" width="800"/>
</p>

For questions, please contact Emma Lejeune `elejeune@bu.edu`
