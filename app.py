from flask import Flask, request, render_template
import a2_tester as predictor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        text_to_predict = request.form.get('text_to_predict')
        if not text_to_predict:
            return render_template('templates/home.html', error='Please enter some text to predict')
        try:
            text_to_predict = text_to_predict.split('\n')
            text_to_predict = list(map(lambda x: x.replace('\r', ''), text_to_predict))
            print(text_to_predict)  
            analysis = predictor.driver(text_to_predict)[0] #here, select among LR, BNB, SVC by index!
            texts = analysis['text'].values
            sentiments = analysis['sentiment'].values
            print(texts)
            print(sentiments)
            return render_template('home.html', texts=texts, sentiments=sentiments)
        except Exception as e:
            print(str(e))
            return render_template('home.html', error='An error occurred during prediction')
    else:
        return render_template('home.html')
    

@app.route('/email', methods=['GET'])
def email():
    return render_template('email.html')
    
if __name__ == '__main__':
    app.run()