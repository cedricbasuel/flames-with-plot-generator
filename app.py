from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
import flames
import flames_with_generate_plot
import sys, yaml


# replace with flask config!!
CONFIG_FILE = sys.argv[1]

with open(CONFIG_FILE) as cfg:
    config = yaml.safe_load(cfg)

app = Flask(__name__)

story_generator = flames_with_generate_plot.load_gpt_model(
        model_path=config['gpt_generate']['dir'], 
        tokenizer_path=config['gpt_generate']['dir'],
        device=0
        )

@app.route('/')
def home():
    return render_template('home1.html')



@app.route('/predict', methods=['POST'])
def predict():

    name1 = request.form['name1']
    name2 = request.form['name2']

    unique_letters = flames.remove_common_letters(name1, name2)
    flames_status = flames.get_flames_status(unique_letters)

    input_prompts = flames_with_generate_plot.create_input_prompts(
        name1, 
        name2, 
        flames_status, 
        n_plots=config['gpt_generate']['n_plots']
        )

    plots = flames_with_generate_plot.generate_plot(
        story_generator,
        input_prompts=input_prompts,
        temperatures=config['gpt_generate']['temperatures'],
        max_length=config['gpt_generate']['max_length'],
        do_sample=config['gpt_generate']['do_sample'],
        top_p=config['gpt_generate']['top_p'],
        top_k=config['gpt_generate']['top_k'],
        repetition_penalty=config['gpt_generate']['rep_penalty'],
        num_return_sequences=config['gpt_generate']['num_return_sequences'],
        )

    plots_list = [plot['generated_text'].replace('<BOS> ','') for plot in plots]

    return render_template('home1.html', stories=plots_list, status=flames_status)



if __name__ == "__main__" :
    app.run(debug=True)