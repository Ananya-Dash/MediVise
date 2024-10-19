from flask import Flask, render_template, request, redirect, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, UserMixin, logout_user
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset for disease prediction
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data preparation and encoding for disease prediction
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Label encoding the prognosis (disease)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Train models for disease prediction
svc = SVC(probability=True)
nb = GaussianNB()
rf = RandomForestClassifier()

svc.fit(X_train, y_train)
nb.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Flask app setup
db = SQLAlchemy()
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///my.db"
app.config["SECRET_KEY"] = 'thisissecret'
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True, nullable=False)
    email = db.Column(db.String, nullable=False)
    password = db.Column(db.String, nullable=False)
    fname = db.Column(db.String, nullable=False)
    lname = db.Column(db.String, nullable=False)
    def __repr__(self):
        return '<User %r' % self.username

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Logout route
@app.route('/logout')
def logout():
    return render_template('logout.html')

# Register route
@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        username = request.form.get('username')
        fname = request.form.get('fname')
        lname = request.form.get('lname')
        user = User(username=username, email=email, password=password, fname=fname, lname=lname)
        db.session.add(user)
        db.session.commit()
        flash('User has been successfully registered', 'success')
        return redirect('/signin')

    return render_template('register.html')

# Signin route
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and password == user.password:
            login_user(user)
            return redirect('/home')
        else:
            flash('Invalid Credentials. Try again', 'danger')
            return redirect('/signin')

    return render_template('signin.html')

# Home after login
@app.route('/home', methods=['POST', 'GET'])
def home():
    return render_template('home.html')

# Disease prediction route
@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        # Collecting input symptoms from the form as binary (1 if checked, 0 if not)
        symptoms = [1 if request.form.get(f'symptom{i}') else 0 for i in range(1, 6)]
        symptoms_array = np.array([symptoms])  # Convert to 2D array for prediction

        # Predict using the three models
        svc_pred = svc.predict(symptoms_array)
        nb_pred = nb.predict(symptoms_array)
        rf_pred = rf.predict(symptoms_array)

        # Combine predictions (ensemble method)
        final_pred = np.bincount([svc_pred[0], nb_pred[0], rf_pred[0]]).argmax()

        disease = le.inverse_transform([final_pred])[0]
        return render_template('disease.html', prediction_text=f'Predicted Disease: {disease}')
    
    return render_template('disease.html')

# Quiz route with health practices prediction
@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    # Load health practices model and label encoder
    model = joblib.load('health_practices_model.pkl')
    le = joblib.load('label_encoder.pkl')

    if request.method == 'POST':
        # Get the user inputs from the quiz form
        water = float(request.form['water'])  # Amount of water in liters
        exercise = float(request.form['exercise'])  # Hours of exercise
        sitting = float(request.form['sitting'])  # Hours spent sitting
        sleep = float(request.form['sleep'])  # Hours of sleep
        diet = float(request.form['diet'])  # Servings of fruits and vegetables
        stress = int(request.form['stress'])  # Stress level on a scale of 1 to 10

        # Prepare input for prediction
        input_data = np.array([[water, exercise, sitting, sleep, diet, stress]])
        predicted_tip_index = model.predict(input_data)[0]
        predicted_tip = le.inverse_transform([predicted_tip_index])[0]

        # Initialize recommendations list
        recommendations = []

        # Severity-based recommendations
        if water < 1.5:
            recommendations.append("You're drinking less than the recommended amount of water. Consider increasing your daily intake by 0.5 liters to stay hydrated.")
        if exercise < 2.5:
            recommendations.append("You’re doing less exercise than recommended. Aim to increase your activity level by 1 hour per week.")
        if sleep < 7:
            recommendations.append("You're getting less sleep than recommended. Try to increase your sleep duration gradually.")
        if stress > 7:
            recommendations.append("Your stress levels are high. Implement stress management techniques like meditation or yoga.")

        # Personalized health goals
        if exercise < 2.0:
            recommendations.append("Set a goal to gradually increase your exercise to at least 2 hours per week over the next month.")
        if water < 2.0:
            recommendations.append("Try to aim for 2 liters of water per day. Gradually increase your intake to reach this goal.")
        if sleep < 7:
            recommendations.append("Aim to increase your sleep to at least 7 hours per night over the next two weeks.")

        # Detailed feedback
        detailed_feedback = []
        detailed_feedback.append("Drinking enough water is crucial for maintaining bodily functions, including temperature regulation and joint lubrication.")
        detailed_feedback.append("Regular exercise can help prevent chronic diseases, improve mood, and enhance overall health.")
        detailed_feedback.append("Adequate sleep is essential for cognitive function, mood regulation, and physical health.")
        detailed_feedback.append("Managing stress is important for mental health, and can help improve your overall well-being.")

        return render_template('quiz.html', tips=[predicted_tip], recommendations=recommendations, detailed_feedback=detailed_feedback)

    return render_template('quiz.html')


# Clinic search route
@app.route('/clinic', methods=['GET', 'POST'])
def clinic():
    # Cities, areas within the cities, and clinic details
    city_data = {
    'Mumbai': {
        'Andheri': [
            {'name': 'Lilavati Hospital', 'address': 'A-301, 3rd Floor, Bandra, Mumbai, Maharashtra 400050'},
            {'name': 'Nanavati Hospital', 'address': 'S.V. Road, Vile Parle West, Mumbai, Maharashtra 400056'},
        ],
        'Bandra': [
            {'name': 'KEM Hospital', 'address': 'A, 1st Floor, Acharya Donde Marg, Parel, Mumbai, Maharashtra 400012'},
            {'name': 'Hinduja Hospital', 'address': 'Veer Savarkar Marg, Mahim, Mumbai, Maharashtra 400016'},
        ],
    },
    'Delhi': {
        'Saket': [
            {'name': 'AIIMS Delhi', 'address': 'Ansari Nagar, New Delhi, Delhi 110029'},
            {'name': 'Fortis Escorts', 'address': 'Okhla Rd, New Delhi, Delhi 110025'},
        ],
        'Karol Bagh': [
            {'name': 'BLK Super Speciality', 'address': 'Pusa Rd, Rajendra Place, New Delhi, Delhi 110005'},
            {'name': 'Sir Ganga Ram Hospital', 'address': 'Rajinder Nagar, New Delhi, Delhi 110060'},
        ],
    },
    'Bangalore': {
        'Whitefield': [
            {'name': 'Narayana Health', 'address': 'Whitefield, Bangalore, Karnataka 560066'},
            {'name': 'Manipal Hospital', 'address': 'Old Airport Road, Bangalore, Karnataka 560017'},
        ],
        'Koramangala': [
            {'name': 'Apollo Hospital', 'address': '154/11, 1st Cross, Koramangala 8th Block, Bangalore, Karnataka 560095'},
            {'name': 'St. John’s Medical College', 'address': 'St. John’s National Academy of Health Sciences, Bangalore, Karnataka 560034'},
        ],
    },
    'Chennai': {
        'T. Nagar': [
            {'name': 'Apollo Hospital', 'address': 'No. 21, Greams Lane, Off Greams Road, Chennai, Tamil Nadu 600006'},
            {'name': 'Fortis Malar', 'address': 'New 12, Old 2, 2nd Avenue, Besant Nagar, Chennai, Tamil Nadu 600090'},
        ],
        'Egmore': [
            {'name': 'Stanley Medical College', 'address': 'Old Jail Road, Esplanade, Chennai, Tamil Nadu 600001'},
            {'name': 'Madras Medical Mission', 'address': '4, Bheema Rao B, Jafferkhanpet, Chennai, Tamil Nadu 600083'},
        ],
    },
    'Kolkata': {
        'Park Street': [
            {'name': 'Apollo Gleneagles', 'address': '58, Canal Circular Road, Kolkata, West Bengal 700054'},
            {'name': 'Fortis Hospitals', 'address': 'L 2, B, 54, Kankurgachi, Kolkata, West Bengal 700054'},
        ],
        'Salt Lake': [
            {'name': 'Amri Hospital', 'address': '52, D, Baisakhi, Salt Lake City, Kolkata, West Bengal 700091'},
            {'name': 'Westbank Hospital', 'address': '1/8, Raja S.C. Mullick Road, Tollygunge, Kolkata, West Bengal 700033'},
        ],
    },
    'Hyderabad': {
        'Banjara Hills': [
            {'name': 'Care Hospital', 'address': 'Road No 1, Banjara Hills, Hyderabad, Telangana 500034'},
            {'name': 'Apollo Hospital', 'address': 'Road No. 3, Banjara Hills, Hyderabad, Telangana 500034'},
        ],
        'Hitech City': [
            {'name': 'Maxcure Hospital', 'address': 'Madhapur, Hitech City, Hyderabad, Telangana 500081'},
            {'name': 'Yashoda Hospital', 'address': 'Hitec City Main Rd, Near Cyber Towers, Madhapur, Hyderabad, Telangana 500081'},
        ],
    },
    'Ahmedabad': {
        'Vastrapur': [
            {'name': 'Zydus Hospital', 'address': 'Zydus Hospital Rd, Thaltej, Ahmedabad, Gujarat 380054'},
            {'name': 'Apollo Hospital', 'address': 'Bopal, Ahmedabad, Gujarat 380058'},
        ],
        'Maninagar': [
            {'name': 'Kiran Hospital', 'address': 'Shahibaug, Ahmedabad, Gujarat 380004'},
            {'name': 'Sanjivani Hospital', 'address': 'Maninagar, Ahmedabad, Gujarat 380008'},
        ],
    },
    'Pune': {
        'Kalyani Nagar': [
            {'name': 'Sahyadri Hospital', 'address': 'Sahyadri Super Specialty Hospital, Nagar Road, Pune, Maharashtra 411014'},
            {'name': 'Ruby Hall Clinic', 'address': 'Kalyani Nagar, Pune, Maharashtra 411006'},
        ],
        'Viman Nagar': [
            {'name': 'KEM Hospital', 'address': 'Viman Nagar, Pune, Maharashtra 411014'},
            {'name': 'Aditya Birla Memorial Hospital', 'address': 'Chinchwad, Pune, Maharashtra 411033'},
        ],
    },
    'Jaipur': {
        'Malviya Nagar': [
            {'name': 'Fortis Hospital', 'address': 'Malviya Nagar, Jaipur, Rajasthan 302017'},
            {'name': 'Manipal Hospital', 'address': 'Sikhar Road, Jaipur, Rajasthan 302028'},
        ],
        'Bani Park': [
            {'name': 'Sawai Man Singh Hospital', 'address': 'Jaipur, Rajasthan 302004'},
            {'name': 'JLN Hospital', 'address': 'Jai Narayan Vyas Colony, Jaipur, Rajasthan 302006'},
        ],
    },
    'Lucknow': {
        'Gomti Nagar': [
            {'name': 'King George\'s Medical University', 'address': 'Shah Mina Road, Lucknow, Uttar Pradesh 226003'},
            {'name': 'Sanjivani Hospital', 'address': 'Gomti Nagar, Lucknow, Uttar Pradesh 226010'},
        ],
        'Alambagh': [
            {'name': 'Dr. Ram Manohar Lohia Institute of Medical Sciences', 'address': 'Lucknow, Uttar Pradesh 226012'},
            {'name': 'Mahatma Gandhi Memorial Medical College', 'address': 'Lucknow, Uttar Pradesh 226018'},
        ],
    },
    'Coimbatore': {
        'RS Puram': [
            {'name': 'KGS Hospital', 'address': '2/145, RS Puram, Coimbatore, Tamil Nadu 641002'},
            {'name': 'Ganga Hospital', 'address': '16, Vellakinaru, Coimbatore, Tamil Nadu 641014'},
        ],
        'Peelamedu': [
            {'name': 'PSG Hospitals', 'address': 'PSG College of Technology, Peelamedu, Coimbatore, Tamil Nadu 641004'},
            {'name': 'C.S.I. Hospital', 'address': 'Kumarasamy Layout, Coimbatore, Tamil Nadu 641008'},
        ],
    },
    'Indore': {
        'Sukhliya': [
            {'name': 'Bombay Hospital', 'address': 'Bombay Hospital Road, Indore, Madhya Pradesh 452001'},
            {'name': 'Indore Hospital', 'address': 'Panchsheel Nagar, Indore, Madhya Pradesh 452001'},
        ],
        'Vijay Nagar': [
            {'name': 'Sanjeevani Hospital', 'address': 'Vijay Nagar, Indore, Madhya Pradesh 452010'},
            {'name': 'Apex Hospital', 'address': 'Sukhlia, Indore, Madhya Pradesh 452010'},
        ],
    },
    'Nagpur': {
        'Civil Lines': [
            {'name': 'Wockhardt Hospital', 'address': 'Sukhlal Bhatia Marg, Civil Lines, Nagpur, Maharashtra 440001'},
            {'name': 'Care Hospital', 'address': 'Maharashtra, Nagpur, Maharashtra 440013'},
        ],
        'Sadar': [
            {'name': 'N M Wadia Hospital', 'address': 'Sadar, Nagpur, Maharashtra 440001'},
            {'name': 'Krishna Hospital', 'address': 'Sadar, Nagpur, Maharashtra 440001'},
        ],
    },
    'Vadodara': {
        'Alkapuri': [
            {'name': 'Pramukh Swami Medical College', 'address': 'Dharampur, Vadodara, Gujarat 390014'},
            {'name': 'Sterling Hospital', 'address': 'Off, Old Padra Rd, Alkapuri, Vadodara, Gujarat 390007'},
        ],
        'Fatehgunj': [
            {'name': 'Shreeji Hospital', 'address': 'Fatehgunj, Vadodara, Gujarat 390002'},
            {'name': 'Vishva Hospital', 'address': 'Sardar Patel Ring Rd, Fatehgunj, Vadodara, Gujarat 390002'},
        ],
    },
    'Surat': {
        'Adajan': [
            {'name': 'Kiran Hospital', 'address': 'Adajan, Surat, Gujarat 395009'},
            {'name': 'Fortis Hospital', 'address': 'Khatodara, Surat, Gujarat 395001'},
        ],
        'Vesu': [
            {'name': 'Noble Hospital', 'address': 'Vesu, Surat, Gujarat 395007'},
            {'name': 'Sunshine Global Hospital', 'address': 'Vesu, Surat, Gujarat 395007'},
        ],
    },
    'Nashik': {
        'Gangapur': [
            {'name': 'Smt. Kashibai Navale Medical College', 'address': 'Gangapur Road, Nashik, Maharashtra 422005'},
            {'name': 'Shree Hospital', 'address': 'Rane Nagar, Nashik, Maharashtra 422003'},
        ],
        'Sinnar': [
            {'name': 'Yashwant Hospital', 'address': 'Sinnar, Nashik, Maharashtra 422103'},
            {'name': 'Nashik City Hospital', 'address': 'Nashik, Maharashtra 422101'},
        ],
    },
    'Patna': {
        'Boring Road': [
            {'name': 'Indira Gandhi Institute of Medical Sciences', 'address': 'Boring Rd, Patna, Bihar 800013'},
            {'name': 'Paras HMRI Hospital', 'address': 'Raja Bazar, Patna, Bihar 800014'},
        ],
        'Patliputra': [
            {'name': 'Anand Medical College', 'address': 'Patliputra, Patna, Bihar 800013'},
            {'name': 'Patna Medical College', 'address': 'Patna, Bihar 800004'},
        ],
    },
    'Dehradun': {
        'Rajpur Road': [
            {'name': 'Max Super Specialty Hospital', 'address': 'Rajpur Road, Dehradun, Uttarakhand 248001'},
            {'name': 'Himalayan Hospital', 'address': 'Dehradun, Uttarakhand 248140'},
        ],
        'Malsi': [
            {'name': 'Yashoda Super Speciality Hospital', 'address': 'Malsi, Dehradun, Uttarakhand 248007'},
            {'name': 'Dr. Yashwant Singh Parmar Medical College', 'address': 'Dehradun, Uttarakhand 248001'},
        ],
    },
    'Chandigarh': {
        'Sector 17': [
            {'name': 'Post Graduate Institute of Medical Education and Research', 'address': 'Sector 12, Chandigarh 160012'},
            {'name': 'Fortis Hospital', 'address': 'Sector 21, Chandigarh 160022'},
        ],
        'Sector 22': [
            {'name': 'Government Medical College and Hospital', 'address': 'Sector 32, Chandigarh 160030'},
            {'name': 'Indira Gandhi Institute of Medical Sciences', 'address': 'Sector 14, Chandigarh 160014'},
        ],
    },
    'Ranchi': {
        'Hinjili': [
            {'name': 'Rajendra Institute of Medical Sciences', 'address': 'Ranchi, Jharkhand 834009'},
            {'name': 'Medanta Hospital', 'address': 'Hinjili, Ranchi, Jharkhand 834002'},
        ],
        'Doranda': [
            {'name': 'Apollo Hospitals', 'address': 'Doranda, Ranchi, Jharkhand 834002'},
            {'name': 'Bansal Hospital', 'address': 'Doranda, Ranchi, Jharkhand 834001'},
        ],
    },
    }


    cities = city_data.keys()
    selected_city = None
    selected_area = None
    areas = None
    clinics = []

    if request.method == 'POST':
        # Check if user has selected a city
        selected_city = request.form.get('city')
        
        # If city is selected, show area options
        if selected_city:
            areas = city_data[selected_city].keys()  # Load areas for the selected city

            # Check if the area is selected for a second submission
            selected_area = request.form.get('area', '')
            if selected_area:
                clinics = city_data[selected_city].get(selected_area, [])  # Get clinics for selected area
    
    return render_template('clinic.html', cities=cities, areas=areas, clinics=clinics, selected_city=selected_city, selected_area=selected_area)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)