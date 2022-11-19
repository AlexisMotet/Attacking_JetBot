from parse import parse
import matplotlib.pyplot as plt
import numpy as np

def parse_results(path):
    results = {}
    dates = {}
    last_id = None
    with open(path, 'r') as f :
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            elif 'batch' in line :
                assert last_id is not None
                results[last_id].append(float(parse('batch {} success rate {}', line)[1]))
            elif 'id' in line :
                results[line] = []
                last_id = line
            else :
                dates[last_id] = line
    return results, dates

def plot_results(training_results, training_dates, test_results):
    _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    assert len(training_results.values()) == len(test_results.values())
    n_curves = len(training_results.values())
    colors = plt.cm.rainbow(np.linspace(0, 1, n_curves))
    for results, ax, title in zip([training_results, test_results], [ax1, ax2], ['train', 'test']) :
        c = 0
        for id, success_rates in results.items():
            ax.plot(range(len(success_rates)), success_rates, label=training_dates[id], c=colors[c])
            c+=1
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        ax.set_title(title)
    plt.show()
    
'''            
if __name__=='__main__' :
    path_training_results = 'C:\\Users\\alexi\\PROJET_3A\\projet_CAS\\training_results.txt'
    path_test_results = 'C:\\Users\\alexi\\PROJET_3A\\projet_CAS\\test_results.txt'
    r_train, training_dates = parse_results(path_test_results)
    r_test, _ = parse_results(path_test_results)
    plot_results(r_train, training_dates, r_test)
'''