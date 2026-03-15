from flask import Flask, request, jsonify, render_template
import util
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "server/uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_label():

    image_file = request.files["image"]
    filename = secure_filename(image_file.filename)

    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(saved_path)

    leaf_label, leaf_conf = util.check_leaf_image(saved_path)

    if leaf_label != "Leaf":
        return jsonify({
            "disease": "Invalid Image",
            "confidence": "0%",
            "advice": "Please upload a clear image of a single leaf."
        })

    prediction = util.get_prediction(saved_path)

    if isinstance(prediction, tuple):
        disease = prediction[0]
        confidence = round(float(prediction[1]) * 100, 2)
    else:
        disease = str(prediction)
        confidence = 0.0

    expert_solutions = {
        "Tomato__Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants immediately. Control whiteflies using insecticides or neem oil.",
        "Tomato__Tomato_mosaic_virus": "Remove infected plants. Disinfect tools and avoid handling plants when wet.",
        "Tomato__Target_Spot": "Apply fungicide and improve air circulation. Avoid overhead watering.",
        "Tomato__Spider_mites Two-spotted_spider_mite": "Spray water on leaf undersides and apply miticides or insecticidal soap.",
        "Tomato__Septoria_leaf_spot": "Remove infected leaves. Apply copper-based fungicide.",
        "Tomato__Leaf_Mold": "Improve ventilation and apply appropriate fungicide.",
        "Tomato__Late_blight": "Destroy infected plants. Apply systemic fungicides and avoid leaf wetness.",
        "Tomato__healthy": "Plant is healthy. Maintain proper watering and sunlight.",
        "Tomato__Early_blight": "Apply copper-based fungicide weekly. Remove infected leaves.",
        "Tomato__Bacterial_spot": "Remove infected leaves. Use copper sprays and disease-free seeds.",
        "Strawberry__Leaf_scorch": "Remove infected leaves and apply fungicide. Avoid overcrowding.",
        "Strawberry__healthy": "Plant is healthy. Maintain proper irrigation and soil health.",
        "Squash__Powdery_mildew": "Apply sulfur or potassium bicarbonate fungicide. Improve air circulation.",
        "Soybean__healthy": "Crop is healthy. Maintain proper fertilization and irrigation.",
        "Raspberry__healthy": "Plant is healthy. Continue proper pruning and watering.",
        "Potato__Late_blight": "Use certified seeds and apply fungicide preventively.",
        "Potato__healthy": "Plant is healthy. Maintain balanced nutrition.",
        "Potato__Early_blight": "Remove infected leaves and apply fungicide early in the season.",
        "Pepper,_bell__healthy": "Plant is healthy. Maintain consistent watering and sunlight.",
        "Pepper,_bell__Bacterial_spot": "Remove infected leaves. Apply copper-based bactericide.",
        "Peach__healthy": "Tree is healthy. Maintain proper pruning and fertilization.",
        "Peach__Bacterial_spot": "Apply copper sprays and remove infected plant debris.",
        "Orange__Haunglongbing_(Citrus_greening)": "Remove infected trees immediately. Control psyllid insects aggressively.",
        "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicide and prune vines to improve airflow.",
        "Grape__healthy": "Vine is healthy. Maintain pruning and disease monitoring.",
        "Grape__Esca_(Black_Measles)": "Remove infected wood and apply protective fungicides.",
        "Grape__Black_rot": "Remove mummified fruits and apply fungicide regularly.",
        "Corn_(maize)__Northern_Leaf_Blight": "Apply fungicide and use resistant hybrids.",
        "Corn_(maize)__healthy": "Crop is healthy. Maintain proper fertilization.",
        "Corn_(maize)__Common_rust_": "Apply fungicide if severe. Plant resistant varieties.",
        "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot": "Use resistant hybrids and apply fungicide if needed.",
        "Cherry_(including_sour)__Powdery_mildew": "Apply fungicide and prune affected branches.",
        "Cherry_(including_sour)__healthy": "Tree is healthy. Maintain proper pruning and watering.",
        "Blueberry__healthy": "Plant is healthy. Maintain acidic soil and proper irrigation.",
        "Apple__healthy": "Tree is healthy. Continue seasonal pruning and care.",
        "Apple__Cedar_apple_rust": "Apply fungicide early in the season. Remove nearby juniper hosts.",
        "Apple__Black_rot": "Remove infected fruit and apply fungicide.",
        "Apple__Apple_scab": "Apply preventive fungicide and remove fallen leaves."
    }

    disease = disease.strip()
    normalized_disease = disease.replace("___", "__")

    if normalized_disease in expert_solutions:
        advice = expert_solutions[normalized_disease]
    elif disease in expert_solutions:
        advice = expert_solutions[disease]
    else:
        advice = "No specific advice available for this disease."

    response = jsonify({
        "disease": disease,
        "confidence": f"{confidence}%",
        "advice": advice
    })

    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    print("Starting Flask server for Plant Disease Prediction...")
    util.load_saved_artifacts()
    util.load_leaf_model()
    app.run(debug=True)