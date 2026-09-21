from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def range_score(value, low, high):
    if low <= value <= high:
        return 100

    distance = low - value if value < low else value - high
    width = max(high - low, 1)

    return clamp(100 - (35 * distance / width))


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        # Get values from webpage
        ph = float(request.form["ph"])
        nitrogen = float(request.form["nitrogen"])
        phosphorus = float(request.form["phosphorus"])
        potassium = float(request.form["potassium"])
        moisture = float(request.form["moisture"])
        ec = float(request.form["ec"])
        temperature = float(request.form["temperature"])
        rainfall = float(request.form["rainfall"])

        # -----------------------------
        # SOIL PARAMETER SCORES
        # -----------------------------

        ph_score = clamp(
            100 - abs(ph - 6.7) * 22
        )

        nitrogen_score = clamp(nitrogen)

        phosphorus_score = clamp(
            phosphorus / 0.55
        )

        potassium_score = clamp(
            potassium / 1.2
        )

        moisture_score = range_score(
            moisture, 20, 50
        )

        ec_score = clamp(
            100 - max(0, ec - 0.8) * 38
        )

        temperature_score = range_score(
            temperature, 18, 32
        )

        rainfall_score = range_score(
            rainfall, 500, 1200
        )

        # -----------------------------
        # SOIL HEALTH
        # -----------------------------

        soil_health = (
            ph_score * 0.16 +
            nitrogen_score * 0.14 +
            phosphorus_score * 0.10 +
            potassium_score * 0.12 +
            moisture_score * 0.14 +
            ec_score * 0.10 +
            temperature_score * 0.08 +
            rainfall_score * 0.06 +
            75 * 0.10
        )

        soil_health = round(
            clamp(soil_health), 1
        )

        # -----------------------------
        # CONDITION
        # -----------------------------

        if soil_health >= 75:
            condition = "Healthy"

        elif soil_health >= 55:
            condition = "Moderate"

        else:
            condition = "Needs Improvement"

        # -----------------------------
        # CROP SUITABILITY
        # -----------------------------

        crops = {}

        crops["Wheat"] = round((
            range_score(ph, 5.5, 7.5) +
            range_score(nitrogen, 40, 100) +
            range_score(phosphorus, 20, 60) +
            range_score(potassium, 30, 100) +
            range_score(moisture, 20, 45) +
            range_score(temperature, 15, 30) +
            range_score(rainfall, 300, 900)
        ) / 7, 1)

        crops["Maize"] = round((
            range_score(ph, 5.5, 7.5) +
            range_score(nitrogen, 50, 120) +
            range_score(phosphorus, 25, 70) +
            range_score(potassium, 30, 120) +
            range_score(moisture, 20, 50) +
            range_score(temperature, 18, 32) +
            range_score(rainfall, 500, 1200)
        ) / 7, 1)

        crops["Rice"] = round((
            range_score(ph, 5.0, 7.5) +
            range_score(nitrogen, 40, 120) +
            range_score(phosphorus, 20, 60) +
            range_score(potassium, 30, 120) +
            range_score(moisture, 30, 70) +
            range_score(temperature, 20, 35) +
            range_score(rainfall, 800, 1800)
        ) / 7, 1)

        crops["Cotton"] = round((
            range_score(ph, 5.5, 8.0) +
            range_score(nitrogen, 40, 100) +
            range_score(phosphorus, 20, 60) +
            range_score(potassium, 40, 120) +
            range_score(moisture, 20, 50) +
            range_score(temperature, 20, 35) +
            range_score(rainfall, 500, 1100)
        ) / 7, 1)

        recommended_crop = max(
            crops,
            key=crops.get
        )

        # -----------------------------
        # EXPLAINABLE AI DEMO
        # -----------------------------

        explanations = {
            "Visual Soil Features": 72,
            "Nitrogen": round(clamp(100 - nitrogen)),
            "Moisture": round(clamp(abs(moisture - 35) * 2)),
            "pH": round(clamp(abs(ph - 6.7) * 8)),
            "EC": round(clamp(max(0, (ec - 0.8) * 35)))
        }

        return jsonify({
            "soil_health": soil_health,
            "condition": condition,
            "recommended_crop": recommended_crop,
            "crop_scores": crops,
            "explanations": explanations
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)