import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
import dice_ml

from libs.cft_to_info import cft_to_info


def generate_counterfactual(x_train, x_val, x_test, y_train, y_val, y_test,
                            model, conterfactual_number=1, offset_min=100, offset_max=200):
    print("[+] Generating Counterfactuals..")
    index_to_map = [str(n) for n in range(x_train.shape[1])]
    df = pd.DataFrame(x_train, columns=index_to_map)
    df_test = pd.DataFrame(x_test, columns=index_to_map)

    ypred = model.predict(df_test)
    ypred = np.argmax(ypred, axis=-1)

    print(metrics.classification_report(y_test, ypred))
    print(metrics.confusion_matrix(y_test, ypred))

    positive = np.where(ypred == 1)[0]
    x_test_true_positive = []
    for p in positive:
        xt = x_test[p]
        if y_test[p] == 1:
            x_test_true_positive.append(xt)

    x_test_true_positive = np.array(x_test_true_positive)
    df_test_true_positive = pd.DataFrame(x_test_true_positive, columns=index_to_map)

    permitted_range = []

    for i in range(offset_min, offset_max):
        row = {}
        for k in df_test_true_positive.columns[0:256]:
            row[k] = [df_test_true_positive.loc[i, k], 1]
        permitted_range.append(row)

    df_with_label = df
    df_with_label["y"] = y_train

    d = dice_ml.Data(dataframe=df_with_label, outcome_name='y', continuous_features=index_to_map)
    m = dice_ml.Model(model=model, backend="TF2")

    exp = dice_ml.Dice(d, m, method="random")

    res = {}

    for i in range(offset_min, offset_max):
        res[i] = {}
        for j in range(conterfactual_number):
            try:
                starting = pd.datetime.datetime.now()
                e1 = exp.generate_counterfactuals(df_test_true_positive.iloc[[i]], total_CFs=1,
                                                  desired_class="opposite",
                                                  features_to_vary=index_to_map[0:256], random_seed=1,
                                                  permitted_range=permitted_range[i - offset_min])
                e1.visualize_as_dataframe(show_only_changes=True)
                res[i] = [e1, (pd.datetime.datetime.now() - starting).seconds]
                break
            except Exception as e:
                print(e)

    res_parsed = {}
    for i in res.keys():
        found, not_found, differences, differences_index, output, test = cft_to_info(res[i][0])
        res_parsed[i] = [found, not_found, differences, differences_index, output, test, res[i][1]]
    with open("conterfactuals.pickle", 'wb') as h:
        pickle.dump(res_parsed, h)

    return res_parsed
