from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json
import csv


rfc = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():

    return render_template('index.html')


def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input =np.zeros(80)
    # set the numerical input as they are
    enc_input[0] = data['Percentage']
    enc_input[1] = data['Follow_up']
    if data['Handle_by'] == 'Vidhya':
        y = 5
    elif data['Handle_by'] =='Hitesh':
        y = 1
    elif data['Handle_by'] =='Shwetambari':
        y = 4
    elif data['Handle_by'] =='Geeta mam':
        y = 0
    elif data['Handle_by'] =='Nilakshi':
        y = 2
    else:
        y = 3

    enc_input[2] = y
    enc_input[3] = data['Fees']

    departments = ['IT', 'admin', 'engineering', 'finance',
    'information_technology', 'management', 'marketing', 'procurement',
    'product', 'sales', 'support', 'temp']

    cols = ['percentage', 'Follow Up', 'Handle_encoded', 'fees', 'Commerce',
       'School', 'Science', 'Adarsh', 'Amar Kor', 'Asmita', 'B.M.C', 'B.P.E.S',
       'Bright', 'CBSC', 'Cambrige', 'Cosmos', 'Donbosco', 'Friends',
       'Gayatri', 'Gopal Sharma', 'Guru Govind', 'Gurunank', 'H.K. Gidwani',
       'I.D.U.B.S', 'I.E.S', 'Ideal ', 'J.J.Accdamy', 'Jay Bharat', 'Jijamata',
       'K.K.V', 'Kalva', 'L.N', 'MS', 'Marathi Vidyamandir', 'Mohad Urdu',
       'Mohradiya', 'Mount Marriy', 'Mulund High school',
       'Mulund Vidaya mandir', 'Mumbai Public', 'NES', 'Nath Pai', 'Navjeevan',
       'New English', 'New Horizan', 'Nobel Angles', 'Nutan', 'Omegha',
       'Oxfard', 'Parag', 'Pawar', 'Podar', 'Powai School', 'Pragatik',
       'Privat', 'Public', 'Rajaram', 'Rajastan', 'Ramkali', 'Ruparlee',
       'Sabu Siddhi', 'Sahyadri', 'Saraswati', 'Shivaji', 'Siddharth',
       'St. Agrsen', 'St. Joseph', 'St. Xaviers', 'St. frances', 'St.Marry',
       'St.Plus', 'Swami Muktanand', 'T.B hiq', 'Thana school',
       'Vani Vidayalay', 'Village', 'Wamanrao', 'Y.C.S', 'only NEET', 'others']

    # redefine the the user inout to match the column name
    redefinded_user_input = data['School_name']
    # search for the index in columns name list
    School_name_column_index = cols.index(redefinded_user_input)
    #print(mark_column_index)
    # fullfill the found index with 1
    enc_input[School_name_column_index] = 1


    redefinded_user_input_1 = data['Studies']
    # search for the index in columns name list
    Studies_column_index = cols.index(redefinded_user_input_1)
    # fullfill the found index with 1
    enc_input[Studies_column_index] = 1


    return enc_input

@app.route('/api',methods=['POST'])
def get_delay():
    result=request.form
    Percentage = result['Percentage']
    School_name = result['School_name']
    Follow_up = result['Follow_up']
    #mark = result['mark']
    Handle_by = result['Handle_by']
    Fees = result['Fees']
    Studies = result['Studies']
    user_input = {'Percentage':Percentage,'School_name':School_name, 'Follow_up':Follow_up, 'Handle_by':Handle_by,'Fees':Fees,'Studies':Studies}


    #print(user_input)
    a = input_to_one_hot(user_input)
    pred =rfc.predict([a])[0]
    #pred = round(pred, 2)
    """
    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError
        """
    prob_pred = rfc.predict_proba([a]).max()
    stats = 'not get Enrolled' if pred == 0 else 'get Enrolled'

    return json.dumps({'status':stats,'prob_pred':prob_pred});
    #fieldnames = ['avg_monthly_hrs','department','last_evaluation', 'n_projects', 'salary','satisfaction','tenure','status']

    """
    with open('namelist.csv','a') as inFile:
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)
            writer.writerow({'avg_monthly_hrs': avg_monthly_hrs, 'department': department,'last_evaluation':last_evaluation, 'n_projects':n_projects, 'salary':salary,'satisfaction':satisfaction,'tenure':tenure,'status':stats})
    """


    # return render_template('result.html',prediction=price_pred)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
