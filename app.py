from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
import flames

app = Flask(__name__)


prompts = {
    'Friendship': [', who is friends with ', ', in a friendship with ', ', a good friend of '],
    'Love': [', who is in love with ', ' who loves ', ', lover of ', ', the partner of '],
    'Affection': [', who has affection for ', ' who is affectionate with ', ' who has nothing but affection for '],
    'Marriage': [', who is married to ', ' the partner of '],
    'Enemy': [', the enemy of ',  ', who hates ', ' the rival of ', ' who is engaged in a rivalry with '],
    'Sibling': [', the sibling of '],
}

# load trained model
model = GPT2LMHeadModel.from_pretrained(config['gpt_generate']['dir'])

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['gpt_generate']['dir'])

# use transformers's text generation pipeline
story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)



@app.route('/')
def home():



    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':

        name1 = request.form.get()
        name2 = request.form.get()

        unique_letters = flames.remove_common_letters(name1, name2)
        flames_status = flames.get_flames_status(unique_letters)




    return render_template('home.html', stories=stories)




if __name__ == "__main__" :
    app.run(debug=False)