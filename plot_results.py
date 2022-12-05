from parse import parse
import matplotlib.pyplot as plt
import numpy as np


def parse_test_results(path):
    results = {}
    dates = {}
    last_id = None
    with open(path, 'r') as f :
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            elif 'img' in line :
                assert last_id is not None
                success_rate =  float(parse('img {} success rate {}', line)[1])
                results[last_id].append(success_rate)
            elif 'id' in line :
                last_id = line
                results[last_id] = []
            else :
                dates[last_id] = line
    return results, dates

def parse_training_results(path):
    training_results = {}
    validation_results = {}
    dates = {}
    last_id = None
    with open(path, 'r') as f :
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            elif 'id' in line :
                last_id = line
                training_results[last_id] = []
                validation_results[last_id] = []
            elif 'epoch' in line :
                training_results[last_id].append([])
                validation_results[last_id].append([])
            elif 'img' in line :
                assert last_id is not None
                res = parse('img {} success rate {} val rate {}', 
                                    line)
                training_results[last_id][-1].append(float(res[1]))
                validation_results[last_id][-1].append(float(res[2]))
            else :
                dates[last_id] = line
    return training_results, validation_results, dates

def plot_results(train_res, valid_res, train_dates, test_res):
    _, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    assert len(train_res.values()) == len(valid_res.values()) == len(test_res.values())
    n_curves = len(train_res.values())
    colors = plt.cm.rainbow(np.linspace(0, 1, n_curves))
    for results, ax, title in zip([train_res, valid_res, test_res], 
                                  [ax1, ax2, ax3], ['train', 'validation', 'test']) :
        c = 0
        if title =='train' or title == 'validation' :
            for id in results.keys():
                for epoch in results[id] :
                    ax.plot(range(len(epoch)), epoch, 
                    label=train_dates[id], c=colors[c])
                c+=1
        else :
            for id, success_rates in results.items():
                ax.plot(range(len(success_rates)), success_rates, 
                        label=train_dates[id], c=colors[c])
                c+=1
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        ax.set_title(title)
    plt.show()

'''
if __name__=='__main__' :
    path_training_results = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\training_results.txt'
    path_test_results = 'U:\\PROJET_3A\\projet_NOUINOU_MOTET\\test_results.txt'
    r_train, r_valid, training_dates = parse_training_results(path_training_results)
    r_test, _ = parse_test_results(path_test_results)
    plot_results(r_train, r_valid, training_dates, r_test)
'''