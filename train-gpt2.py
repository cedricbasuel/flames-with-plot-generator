import pandas
import random

data = pandas.read_csv('/home/cedric/Downloads/wiki_movie_plots_deduped.csv')
# print(data.columns)
plots_raw =  data.Plot.copy()
plots_raw = list(plots_raw)
plots = ['<BOS> ' + plot + ' <EOS>' for plot in plots_raw]

plots_train_index = random.sample(range(len(plots)), int(0.25 * len(plots)))
plots_text_index = [ind for ind in range(len(plots)) if ind not in plots_train_index]

plots_train = [plots[ind] for ind in plots_train_index]
plots_test = [plots[ind] for ind in plots_text_index]

plot_train_string = ''
for plot in plots_train:
    plot_train_string += plot + '\n'

plot_test_string = ''
for plot in plots_test:
    plot_test_string += plot + '\n'



with open('/home/cedric/Desktop/cv-tf2/plots_train_gpt2.txt', 'w') as f:
    f.write(plot_train_string)

with open('/home/cedric/Desktop/cv-tf2/plots_test_gpt2.txt', 'w') as f:
    f.write(plot_test_string)