from flask import Flask, render_template, request

# Load the trained model from the pickle file
import pickle

with open('wine_quality.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define the route for the home page (form page)
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction result page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the web form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])
    feature5 = float(request.form['feature5'])
    feature6 = float(request.form['feature6'])
    feature7 = float(request.form['feature7'])
    feature8 = float(request.form['feature8'])
    feature9 = float(request.form['feature9'])
    feature10 = float(request.form['feature10'])
    feature11 = float(request.form['feature11'])
    # Add more features as needed

    # Preprocess the input features
    # Make sure to preprocess the input data the same way you did during model training (e.g., normalization)

    # Make the prediction using the loaded model
    prediction = model.predict([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11]])
    
    # Get the predicted class label (0 or 1)
    predicted_label = prediction[0]

    # Print the input features and prediction for debugging
    print("Input features:", [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11])
    print("Predicted Label:", predicted_label)

    # Return the prediction result page
    return render_template('index.html', label=predicted_label)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    