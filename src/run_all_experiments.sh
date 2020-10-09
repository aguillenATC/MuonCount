#python3 split_train_test_val.py
#python3 split_cross_validation.py
#python3 isolate_whole_QGSJETII_test.py

python3 linear_regression.py
echo "Finished Linear regression hyper-tuning and single partition adjustment. 5-fold cross validation results:"
python3 linear_regression_cv.py
python3 poly_regression.py
echo "Finished Poly regression hyper-tuning and single partition adjustment. 5-fold cross validation results:"
python3 poly_regression_cv.py
python3 decision_tree.py
echo "Finished Decision tree hyper-tuning and single partition adjustment. 5-fold cross validation results:"
python3 decision_tree_cv.py
python3 random_forest.py
echo "Finished Random forest hyper-tuning and single partition adjustment. 5-fold cross validation results:"
python3 random_forest_cv.py
python3 gradient_boosting.py
echo "Finished Gradient boosting hyper-tuning and single partition adjustment. 5-fold cross validation results:"
python3 gradient_boosting_cv.py
python3 extreme_gradient_boost.py
echo "Finished XGBoost hyper-tuning and single partition adjustment. 5-fold cross validation results:"
python3 extreme_gradient_boost_cv.py
python3 svm_regression.py
echo "Finished SVM hyper-tuning and single partition adjustment. 5-fold cross validation results:"
python3 svm_regression_cv.py
#python3 mlp_regression.py
#echo "Finished MLP hyper-tuning and single partition adjustment. 5-fold cross validation results:"
#python3 mlp_regression_cv.py
#python3 dnn.py
#echo "Finished DNN hyper-tuning and single partition adjustment. 5-fold cross validation results:"
#python3 dnn_cv.py