from flask import Flask,jsonify,request
import json
import pickle
import numpy as np

app = Flask(__name__)
__locations = None
__area = None
__data_columns = None
__model = None

@app.route('/get_location_structure')
def get_location_structure():
    response = jsonify({
        'location' : load_my_important_data(),
        'area' : load_area()
    })
    response.headers.add('Acess-Control-Allow-origin', '*')
    return response

def load_my_important_data():
    print("Loading data start...")
    global __locations
    global __data_columns
    global __model
    with open('columns.json','r') as file:
        __data_columns = json.load(file)['coloumns_info']
        __locations = __data_columns[6::]
    with open('bengaluru_house_prices_model.pickle','rb') as file:
        __model = pickle.load(file)
    print("Loading saved data done.")
    return __locations

def load_area():
    global __area
    __area = __data_columns[3:6:]
    return __area



def get_estimated_price(area,location,sqft,bhk,bath):
    loc_index_location = __locations.index(location.lower())
    loc_index_area = __area.index(area.lower())
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    x[loc_index_location] = 1
    x[loc_index_area] = 1
    return round(__model.predict([x])[0],2)



if __name__ == '__main__':
    print("Flask started running..")
    app.run()