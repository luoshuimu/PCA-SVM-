import  time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier

# 加载数据  x为128*128个享受点 y为标签
x_train = pd.read_csv('DRdata.csv')
x_test = pd.read_csv('DSdata.csv')
y_train =pd.read_csv('DRsex.csv')
y_test =pd.read_csv('DSsex.csv')
# 归一化处理
x_train = x_train.values / 255
y_train = y_train.values
x_test = x_test.values / 255
y_test = y_test.values

# 定义一个函数，可方便调参
def get_parameter(n,m,x_train,y_train,x_test,y_test):
    # 开始时间
    start_time = time.time()
    # PCA提取特征与降维
    pca = PCA(n_components=n)
    pca.fit(x_train) # 训练
    x_train_pca = pca.transform(x_train)  # 对训练集降维
    x_test_pca = pca.transform(x_test)    # 对测试集降维
    model = SVC(kernel='rbf',C=8,gamma=0.001)  #最佳参数
    '''
    # 建立交叉验证最佳参数
    # 创建一个5次拆分的KFold对象
    folds = KFold(n_splits=5, shuffle=True,random_state = 10)
    #指定超参数范围
    param_grid = {'gamma': [1e-2, 1e-3, 1e-4,1e-5], 'C': np.arange(1, 10, 1).tolist()}
    # SVM 创建模型
    model = SVC(kernel='rbf')
    # 执行网格搜索并计算最高准确率
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=folds, verbose=1,return_train_score=True)
    grid_search.fit(x_train_pca, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    cv_results = pd.DataFrame(grid_search.cv_results_)
    print(cv_results)
    print(best_params)
    print(best_score)
    '''
    model.fit(x_train_pca, y_train)
    '''
    #Bagging-SVM集成算法
    # 基础分类器
    base_classifier = SVC(kernel='rbf', C=8, gamma=0.001)
    model = BaggingClassifier(base_classifier, n_estimators=m)
    model.fit(x_train_pca, y_train)
    # 评估
    '''
    score = model.score(x_test_pca, y_test)
    print('准确率：', score)
    # 结束时间
    end_time = time.time()
    # 打印运行时间
    print('cost: ',end_time-start_time)
    print('-' * 50)
get_parameter(0.999,1,x_train,y_train,x_test,y_test)

