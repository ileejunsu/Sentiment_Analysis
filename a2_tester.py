import pickle
import a0_preprocess as pp
import pandas as pd

def load_models(type):
    
    # Load the vectoriser.
    file = open('vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the Model.
    if type=='LR':
        file = open('Sentiment-LR.pickle', 'rb')
    elif type=='SVC':
        file = open('Sentiment-SVC.pickle', 'rb')
    elif type=='BNB':
        file = open('Sentiment-BNB.pickle', 'rb')
    model = pickle.load(file)
    file.close()
    
    return vectoriser, model

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(pp.preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

def driver(text):
    # Loading the models.
    vectoriser, LR = load_models('LR')
    vectoriser, SVC = load_models('BNB')
    vectoriser, BNB = load_models('SVC')

    df1 = predict(vectoriser, LR, text)
    df2 = predict(vectoriser, BNB, text)
    df3 = predict(vectoriser, SVC, text)
    
    print("LR Model : ")
    print(df1)
    print()
    print("BNB Model : ")
    print(df2)
    print()
    print("SVC Model : ")
    print(df3)
    print()

    return [df1, df2, df3]