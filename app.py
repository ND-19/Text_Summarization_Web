from flask import Flask, redirect, url_for, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
app = Flask(__name__)
 
@app.route('/')
def home():
    return render_template('home.html')
 
@app.route('/predict', methods=['POST'])
def login():
    if request.method == 'POST':
        article = request.form['text']
        tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
        inputs = tokenizer(article, return_tensors="pt").input_ids
        model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")  

        outputs = model.generate(inputs,min_length = 50, max_new_tokens=100, do_sample=False)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        data = summary

        return render_template('home.html', summ = data, article = article.strip())
    
 
 
if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)