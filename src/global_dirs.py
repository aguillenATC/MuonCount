data_dir='../Datos/'
results_dir='../Results/'
##data_folder="2018-04-03-s1000-instead-of-energy/"
data_folder="2018-06-21[2018-03-13]-good-data/"
#data_folder="[LO MEJOR QUE TENGO]2018-06-21[2018-03-13]-good-data/"
#results_folder="02032019-Alberto-VS2-"
results_folder="100fold-Alberto-VS1-"
results_folder_VS2="100fold-Alberto-VS2-"

data_path=data_dir+data_folder
results_path=results_dir+results_folder
results_path_VS2=results_dir+results_folder_VS2

splitted_data_path=data_dir+'splitted_data/'
splitted_data_path_real=data_dir+'splitted_data_REAL/'
cross_val_data_path=data_dir+'splitted_data_cross_validation/'

optimal_models='../optimal_net/'

linear_regression_path=results_path+'linear_regression/'
poly_regression_path=results_path+'poly_regression/'
decision_tree_path=results_path+'decision_tree/'
random_forest_path=results_path+'random_forest/'
gradient_boosting_path=results_path+'gradient_boosting/'
xgboost_path=results_path+'xgboost/'

linear_regression_path_VS2=results_path_VS2+'linear_regression/'
poly_regression_path_VS2=results_path_VS2+'poly_regression/'
decision_tree_path_VS2=results_path_VS2+'decision_tree/'
random_forest_path_VS2=results_path_VS2+'random_forest/'
gradient_boosting_path_VS2=results_path_VS2+'gradient_boosting/'
xgboost_path_VS2=results_path_VS2+'xgboost/'

#gaussian_processes_path=results_path+'gaussian_processes/'
svm_path=results_path+'svm/'
svm_path_VS2=results_path_VS2+'svm/'

mlp_path=results_path+'mlp/'
mlp_path_VS2=results_path_VS2+'mlp/'

dnn_path=results_path+'dnn/'

folds=100

#(list of indexes of variables, Select)
#If select = True, select indexes and discard the rest of variables
#If select = False, discard indexes and select the rest of variables
#variable_selection = ([0,1,2,7,8,10],False)
variable_selection = ([0,1,2,4,7,8],False)
