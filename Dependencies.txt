Dependencies
-scikit-learn: Machine learning library
-pandas: For data manipulation
-numpy: For numerical computations
-matplotlib: For visualization
-seaborn: For advanced visualization
-math, time (base python imports)






Files------------------------------------
SPECTF_test.csv 
SPECTF_train.csv - Normalized test and train datasets


SPECTFtest.csv
SPECTFtrain.csv - Original, non-normalized test and train datasets


metrics.xlsx - Full output of classifier metrics for the manually implemented SVM


Outputs_bayes_libSVM (folder)- graphs from the SKLearn Bayes classifier, contains ROC AUC graphs not included in the paper itself



ML_Final_Project.py - the python file containing the manually implemented SVM. Can be run as is, it only needs the normalized datasets in the same directory. It is written in Python 3.11. The program iterates through all combinations of hyperparameters specified and takes about 17 minutes to complete. The main function can be found at the bottom of the program, and contains the main loop. From the main loop the DualLearner() function is called, which iterates the specified number of times, and calls GradientUpdate() at the end of each iteration. Kernel function implementations are located directly below GradientUpdate(), and they are passed from main (located in the list appropriately named kernels) to the subsequent function calls. Once a set of alphas is learned, the separator w and offset b are retrieved from the functions GetW() and GetB(), and confusion matrix values are counted from GetPerformanceMetrics(). THIS PROGRAM IS WRAPPED IN AN 'if __name__ == '__main__' FUNCTION, the program should run from the command line, otherwise the main function can be commented out. Alternatively, if you are using an IDE, you may set the file to be the startup file, or run it manually from there.




BayesClassifier.py - SKLearn implemented bayes classifier

SVM_SkLearn - SKLearn implemented svm


ML_FinalProject_Report.pdf - the report itself

