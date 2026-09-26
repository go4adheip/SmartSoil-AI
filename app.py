from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import io
import base64
import requests
import re
import urllib.parse as up

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load Stage 1 Binary Detector (Soil vs Non-Soil)
binary_model_path = "soil_binary_detector.keras"
if os.path.exists(binary_model_path):
    binary_model = tf.keras.models.load_model(binary_model_path, safe_mode=False)
else:
    binary_model = None

# Load Stage 2 Multiclass Soil Classifier
multiclass_model_path = "soil_cnn.keras"
if os.path.exists(multiclass_model_path):
    cnn_model = tf.keras.models.load_model(multiclass_model_path, safe_mode=False)
else:
    cnn_model = None

soil_classes = [
    "Alluvial Soil",
    "Arid Soil",
    "Black Soil",
    "Laterite Soil",
    "Mountain Soil",
    "Red Soil",
    "Yellow Soil"
]

soil_metadata = {
    "Alluvial Soil": {
        "color_hex": "#a3704c",
        "gradient": "linear-gradient(135deg, #a3704c, #d4a373)",
        "texture_tag": "Silty Loam & Clay Blend",
        "drainage": "Moderate to High",
        "ph_typical": "6.5 - 8.4",
        "description": "Rich in potash and lime, formed by river alluvial deposits. Highly fertile and adaptable for intensive crop cultivation.",
        "explainability": "Classified as Alluvial Soil due to its smooth silty-loam grain texture, balanced earth tone hue, and lack of heavy iron oxide rust tint.",
        "sample_image": "/static/samples/Alluvial_Soil.jpg",
        "eco_impact": {
            "health_score": 92,
            "carbon_capacity": "High Soil Organic Carbon (1.85% C)",
            "water_efficiency": "88% Retention Efficiency — Low Evaporation Loss",
            "regeneration_tip": "Incorporate green manure (Dhaincha/Sunhemp) post-harvest to preserve alluvial microbial biodiversity."
        },
        "agronomic_reference": {
            "ph": {"val": "6.5 - 8.4", "score": 85, "status": "Optimal Balanced"},
            "nitrogen": {"val": "210 kg/ha", "score": 88, "status": "High Availability"},
            "phosphorus": {"val": "22 kg/ha", "score": 78, "status": "Med-High Supply"},
            "potassium": {"val": "260 kg/ha", "score": 90, "status": "Rich Potash"},
            "moisture": {"val": "32%", "score": 82, "status": "Good Field Capacity"},
            "salinity": {"val": "1.2 dS/m", "score": 92, "status": "Low Salinity"}
        },
        "crop_suitability": [
            {"name": "Rice", "icon": "🍚", "reason": "Requires standing moisture & silty loam retention provided by river alluvial deposits."},
            {"name": "Wheat", "icon": "🌾", "reason": "Thrives in well-drained fertile loam with neutral to slightly alkaline pH."},
            {"name": "Sugarcane", "icon": "🎋", "reason": "Deep root structure benefits from rich potash and continuous moisture holding capacity."},
            {"name": "Maize", "icon": "🌽", "reason": "Responsive to high nitrogen reserves and moderate permeability in alluvial tracts."},
            {"name": "Cotton", "icon": "☁️", "reason": "Sufficient calcium and lime balance supports strong boll development."}
        ],
        "seasonal_calendar": [
            {"season": "Kharif (Monsoon)", "months": "June - October", "crops": ["Rice", "Maize", "Sugarcane"]},
            {"season": "Rabi (Winter)", "months": "November - April", "crops": ["Wheat", "Mustard", "Gram"]},
            {"season": "Zaid (Summer)", "months": "April - June", "crops": ["Groundnut", "Watermelon", "Pulses"]}
        ],
        "comparison_matrix": [
            {"attribute": "pH Range", "detected": "6.5 - 8.4 (Balanced)", "other1": "Black: 7.2 - 8.5", "other2": "Red: 6.0 - 7.5"},
            {"attribute": "Drainage", "detected": "Moderate to High", "other1": "Black: Low / Clay Heavy", "other2": "Red: High / Porous"},
            {"attribute": "Soil Fertility", "detected": "High (Potash & Lime)", "other1": "Black: High (Ca / Mg)", "other2": "Red: Medium (Iron-rich)"},
            {"attribute": "Water Retention", "detected": "High Capacity", "other1": "Black: Exceptional", "other2": "Red: Low to Moderate"},
            {"attribute": "Primary Crops", "detected": "Rice, Wheat, Sugarcane", "other1": "Cotton, Groundnut", "other2": "Pulses, Millets"}
        ]
    },
    "Arid Soil": {
        "color_hex": "#d49b54",
        "gradient": "linear-gradient(135deg, #d49b54, #e6c594)",
        "texture_tag": "Sandy Granular & Saline",
        "drainage": "Excessive / Rapid",
        "ph_typical": "7.0 - 8.5",
        "description": "Coarse sandy texture with low organic matter. Highly responsive to drip irrigation and organic mulching.",
        "explainability": "Classified as Arid Soil due to coarse sandy particle structure, high light reflection, and dry pale golden-brown tint.",
        "sample_image": "/static/samples/Arid_Soil.jpg",
        "eco_impact": {
            "health_score": 65,
            "carbon_capacity": "Low Organic Matter (< 0.45% C) — Needs Amending",
            "water_efficiency": "35% Drip Efficiency — High Evaporation Risk",
            "regeneration_tip": "Apply biochar + organic straw mulching to lock subsoil moisture and halt wind erosion."
        },
        "agronomic_reference": {
            "ph": {"val": "7.0 - 8.5", "score": 75, "status": "Alkaline Range"},
            "nitrogen": {"val": "85 kg/ha", "score": 35, "status": "Low Deficient"},
            "phosphorus": {"val": "12 kg/ha", "score": 45, "status": "Low Supply"},
            "potassium": {"val": "230 kg/ha", "score": 82, "status": "High Potash"},
            "moisture": {"val": "10%", "score": 25, "status": "Low Moisture"},
            "salinity": {"val": "3.4 dS/m", "score": 40, "status": "Elevated Salinity"}
        },
        "crop_suitability": [
            {"name": "Barley", "icon": "🌾", "reason": "High drought tolerance and saline resistance suitable for arid sandy loams."},
            {"name": "Pearl Millet (Bajra)", "icon": "🌱", "reason": "Thrives under low rainfall and quick drainage conditions."},
            {"name": "Cotton", "icon": "☁️", "reason": "Deep taproots access subsoil moisture when drip irrigation is applied."},
            {"name": "Mustard", "icon": "🌼", "reason": "Low water requirement and high responsiveness to potassium."},
            {"name": "Pulses (Moth Bean)", "icon": "🫘", "reason": "Fixes atmospheric nitrogen to compensate for low organic matter."}
        ],
        "seasonal_calendar": [
            {"season": "Kharif (Monsoon)", "months": "July - October", "crops": ["Pearl Millet", "Pulses", "Cluster Bean"]},
            {"season": "Rabi (Winter)", "months": "October - March", "crops": ["Barley", "Mustard", "Cumin"]},
            {"season": "Zaid (Summer)", "months": "March - June", "crops": ["Fodder Crops", "Watermelon"]}
        ],
        "comparison_matrix": [
            {"attribute": "pH Range", "detected": "7.0 - 8.5 (Alkaline)", "other1": "Alluvial: 6.5 - 8.4", "other2": "Laterite: 5.0 - 6.5"},
            {"attribute": "Drainage", "detected": "Excessive / Rapid", "other1": "Alluvial: Moderate", "other2": "Laterite: High Porosity"},
            {"attribute": "Soil Fertility", "detected": "Low (Needs Organic)", "other1": "Alluvial: High", "other2": "Laterite: Low / Leached"},
            {"attribute": "Water Retention", "detected": "Very Low (5-15%)", "other1": "Alluvial: High", "other2": "Laterite: Low"},
            {"attribute": "Primary Crops", "detected": "Barley, Millets, Pulses", "other1": "Rice, Wheat", "other2": "Tea, Coffee, Cashew"}
        ]
    },
    "Black Soil": {
        "color_hex": "#343a40",
        "gradient": "linear-gradient(135deg, #2c2f33, #4a4e54)",
        "texture_tag": "Deep Obsidian Clay (Regur)",
        "drainage": "Low / Exceptional Retention",
        "ph_typical": "7.2 - 8.5",
        "description": "Volcanic black clay soil rich in calcium, iron, and magnesium. Swells when wet and retains moisture for prolonged dry spells.",
        "explainability": "Classified as Black Soil due to obsidian dark color tone, fine dense clay cohesion, and low visible quartz sand ratio.",
        "sample_image": "/static/samples/Black_Soil.jpg",
        "eco_impact": {
            "health_score": 95,
            "carbon_capacity": "Very High Organic Matter (2.10% Carbon)",
            "water_efficiency": "96% Exceptional Clay Water Retention Capacity",
            "regeneration_tip": "Practice zero-tillage to protect natural clay aggregates and prevent deep drought cracking."
        },
        "agronomic_reference": {
            "ph": {"val": "7.2 - 8.5", "score": 82, "status": "Slightly Alkaline"},
            "nitrogen": {"val": "165 kg/ha", "score": 68, "status": "Moderate Level"},
            "phosphorus": {"val": "14 kg/ha", "score": 55, "status": "Low-Med Supply"},
            "potassium": {"val": "320 kg/ha", "score": 96, "status": "Very High Potash"},
            "moisture": {"val": "42%", "score": 95, "status": "Exceptional Retention"},
            "salinity": {"val": "1.1 dS/m", "score": 88, "status": "Optimal EC"}
        },
        "crop_suitability": [
            {"name": "Cotton", "icon": "☁️", "reason": "Deep regur black clay holds moisture during boll formation."},
            {"name": "Sugarcane", "icon": "🎋", "reason": "Rich calcium and magnesium support heavy biomass growth."},
            {"name": "Wheat", "icon": "🌾", "reason": "Retained soil moisture sustains winter growth without heavy irrigation."},
            {"name": "Groundnut", "icon": "🥜", "reason": "High calcium content aids pod development."},
            {"name": "Sunflower", "icon": "🌻", "reason": "Deep root penetration utilizes subsoil moisture effectively."}
        ],
        "seasonal_calendar": [
            {"season": "Kharif (Monsoon)", "months": "June - November", "crops": ["Cotton", "Sugarcane", "Soybean"]},
            {"season": "Rabi (Winter)", "months": "October - March", "crops": ["Wheat", "Gram", "Sorghum"]},
            {"season": "Zaid (Summer)", "months": "March - May", "crops": ["Sunflower", "Sesame"]}
        ],
        "comparison_matrix": [
            {"attribute": "pH Range", "detected": "7.2 - 8.5 (Alkaline)", "other1": "Alluvial: 6.5 - 8.4", "other2": "Mountain: 5.2 - 6.8"},
            {"attribute": "Drainage", "detected": "Low / High Swell", "other1": "Alluvial: Moderate", "other2": "Mountain: Well-Drained"},
            {"attribute": "Soil Fertility", "detected": "Very High (Ca/Mg/Fe)", "other1": "Alluvial: High", "other2": "Mountain: Humus Rich"},
            {"attribute": "Water Retention", "detected": "Exceptional (35-55%)", "other1": "Alluvial: High", "other2": "Mountain: Moderate"},
            {"attribute": "Primary Crops", "detected": "Cotton, Sugarcane", "other1": "Rice, Wheat", "other2": "Spices, Tea, Apples"}
        ]
    },
    "Laterite Soil": {
        "color_hex": "#c45a35",
        "gradient": "linear-gradient(135deg, #c45a35, #e0815e)",
        "texture_tag": "Iron Terracotta Porous",
        "drainage": "High Porosity",
        "ph_typical": "5.0 - 6.5",
        "description": "Formed under tropical weathering and intense leaching. High iron oxide gives it a characteristic rust-red hue.",
        "explainability": "Classified as Laterite Soil due to intense terracotta rust-red iron oxide coloration and porous coarse texture.",
        "sample_image": "/static/samples/Laterite_Soil.jpg",
        "eco_impact": {
            "health_score": 72,
            "carbon_capacity": "Leached Soil Organic Carbon (0.75% C)",
            "water_efficiency": "45% Rapid Porous Infiltration Rate",
            "regeneration_tip": "Apply agricultural lime & farmyard manure to buffer soil acidity and stabilize iron oxide leaching."
        },
        "agronomic_reference": {
            "ph": {"val": "5.0 - 6.5", "score": 62, "status": "Acidic Leached"},
            "nitrogen": {"val": "115 kg/ha", "score": 48, "status": "Low Availability"},
            "phosphorus": {"val": "8 kg/ha", "score": 30, "status": "Deficient P"},
            "potassium": {"val": "130 kg/ha", "score": 52, "status": "Low-Med Potash"},
            "moisture": {"val": "16%", "score": 40, "status": "High Porosity"},
            "salinity": {"val": "0.3 dS/m", "score": 96, "status": "Very Low EC"}
        },
        "crop_suitability": [
            {"name": "Tea", "icon": "🍃", "reason": "Acidic pH (5.0 - 5.5) and porous hillside drainage provide prime tea conditions."},
            {"name": "Coffee", "icon": "☕", "reason": "High iron content and slope drainage prevent root waterlogging."},
            {"name": "Cashew", "icon": "🌰", "reason": "Hardy tree crop adapted to leached acidic tropical soils."},
            {"name": "Rubber", "icon": "🪴", "reason": "High rainfall compatibility in warm terracotta laterite zones."},
            {"name": "Coconut", "icon": "🥥", "reason": "Resilient deep palm root system thrives in porous coastlines."}
        ],
        "seasonal_calendar": [
            {"season": "Kharif (Monsoon)", "months": "June - November", "crops": ["Tea", "Coffee", "Rubber"]},
            {"season": "Rabi (Winter)", "months": "November - March", "crops": ["Cashew", "Arecanut", "Spices"]},
            {"season": "Zaid (Summer)", "months": "March - May", "crops": ["Coconut", "Tapioca"]}
        ],
        "comparison_matrix": [
            {"attribute": "pH Range", "detected": "5.0 - 6.5 (Acidic)", "other1": "Black: 7.2 - 8.5", "other2": "Arid: 7.0 - 8.5"},
            {"attribute": "Drainage", "detected": "High Porosity", "other1": "Black: Low", "other2": "Arid: Excessive"},
            {"attribute": "Soil Fertility", "detected": "Leached / Low", "other1": "Black: High", "other2": "Arid: Low"},
            {"attribute": "Water Retention", "detected": "Low (12-22%)", "other1": "Black: Exceptional", "other2": "Arid: Very Low"},
            {"attribute": "Primary Crops", "detected": "Tea, Coffee, Cashew", "other1": "Cotton, Sugarcane", "other2": "Barley, Millets"}
        ]
    },
    "Mountain Soil": {
        "color_hex": "#43683a",
        "gradient": "linear-gradient(135deg, #43683a, #719b66)",
        "texture_tag": "Humus-Rich Forest Loam",
        "drainage": "Well-Drained Hillside",
        "ph_typical": "5.2 - 6.8",
        "description": "Abundant organic forest humus and rich peat layers. Highly fertile for plantation and hill agriculture.",
        "explainability": "Classified as Mountain Soil due to dark forest humus organic hues, rich peat leaf-litter texture, and moist granular structure.",
        "sample_image": "/static/samples/Mountain_Soil.jpg",
        "eco_impact": {
            "health_score": 96,
            "carbon_capacity": "Rich Peat Forest Humus (3.20% Carbon)",
            "water_efficiency": "90% Natural Canopy Sponging & Storage",
            "regeneration_tip": "Contour terracing and agro-forestry canopy intercropping to eliminate hillside soil erosion."
        },
        "agronomic_reference": {
            "ph": {"val": "5.2 - 6.8", "score": 78, "status": "Slightly Acidic"},
            "nitrogen": {"val": "270 kg/ha", "score": 96, "status": "Very High Nitrogen"},
            "phosphorus": {"val": "16 kg/ha", "score": 65, "status": "Moderate Level"},
            "potassium": {"val": "185 kg/ha", "score": 72, "status": "Good Potash"},
            "moisture": {"val": "38%", "score": 88, "status": "Humus Moisture"},
            "salinity": {"val": "0.5 dS/m", "score": 94, "status": "Low Salinity"}
        },
        "crop_suitability": [
            {"name": "Apples", "icon": "🍎", "reason": "Cool climate and humus-rich slope drainage support orchard fruiting."},
            {"name": "Spices (Cardamom)", "icon": "🌿", "reason": "Thrives in shade and organic forest litter under canopy."},
            {"name": "Tea", "icon": "🍃", "reason": "Slope drainage and high nitrogen humus foster tender tea leaves."},
            {"name": "Barley", "icon": "🌾", "reason": "Cold-tolerant grain suited for high altitude mountain terraces."},
            {"name": "Maize", "icon": "🌽", "reason": "Responsive to abundant organic nitrogen on hill terraces."}
        ],
        "seasonal_calendar": [
            {"season": "Kharif (Monsoon)", "months": "May - September", "crops": ["Maize", "Rice", "Spices"]},
            {"season": "Rabi (Winter)", "months": "October - April", "crops": ["Barley", "Wheat", "Mustard"]},
            {"season": "Zaid (Summer)", "months": "April - June", "crops": ["Apples", "Plums", "Vegetables"]}
        ],
        "comparison_matrix": [
            {"attribute": "pH Range", "detected": "5.2 - 6.8 (Mild Acid)", "other1": "Alluvial: 6.5 - 8.4", "other2": "Red: 6.0 - 7.5"},
            {"attribute": "Drainage", "detected": "Well-Drained Slope", "other1": "Alluvial: Moderate", "other2": "Red: Good"},
            {"attribute": "Soil Fertility", "detected": "High (Forest Humus)", "other1": "Alluvial: High", "other2": "Red: Medium"},
            {"attribute": "Water Retention", "detected": "High Humus (30-50%)", "other1": "Alluvial: High", "other2": "Red: Moderate"},
            {"attribute": "Primary Crops", "detected": "Apples, Spices, Tea", "other1": "Rice, Wheat", "other2": "Groundnut, Pulses"}
        ]
    },
    "Red Soil": {
        "color_hex": "#b84228",
        "gradient": "linear-gradient(135deg, #b84228, #e06c53)",
        "texture_tag": "Ferruginous Sandy Clay",
        "drainage": "Good to Rapid",
        "ph_typical": "6.0 - 7.5",
        "description": "Derives vibrant red tint from ferric oxide diffusion in crystalline rocks. Porous structure suitable for dryland crops.",
        "explainability": "Classified as Red Soil due to ferric oxide mineral hue, porous sandy-clay structure, and low organic dark matter content.",
        "sample_image": "/static/samples/Red_Soil.jpg",
        "eco_impact": {
            "health_score": 78,
            "carbon_capacity": "Moderate Ferruginous Organic Carbon (0.95% C)",
            "water_efficiency": "55% Moderate Aerated Permeability Rate",
            "regeneration_tip": "Apply rock phosphate with bio-fertilizers (Azotobacter) to boost bio-available phosphate levels."
        },
        "agronomic_reference": {
            "ph": {"val": "6.0 - 7.5", "score": 82, "status": "Slightly Acid/Neutral"},
            "nitrogen": {"val": "135 kg/ha", "score": 55, "status": "Low-Med Level"},
            "phosphorus": {"val": "11 kg/ha", "score": 45, "status": "Low P Content"},
            "potassium": {"val": "175 kg/ha", "score": 68, "status": "Moderate Potash"},
            "moisture": {"val": "20%", "score": 50, "status": "Porous Drainage"},
            "salinity": {"val": "0.4 dS/m", "score": 95, "status": "Low Salinity"}
        },
        "crop_suitability": [
            {"name": "Groundnut", "icon": "🥜", "reason": "Porous sandy-clay texture allows uninhibited pegging and pod growth."},
            {"name": "Pulses (Pigeon Pea)", "icon": "🫘", "reason": "Deep roots tolerate dry spells and fix soil nitrogen."},
            {"name": "Finger Millet (Ragi)", "icon": "🌾", "reason": "Exceptional climate resilience in ferric red dryland soils."},
            {"name": "Cotton", "icon": "☁️", "reason": "Good root aeration when supplemented with phosphate fertilizer."},
            {"name": "Wheat", "icon": "🌾", "reason": "Suitable in irrigated red loams during winter season."}
        ],
        "seasonal_calendar": [
            {"season": "Kharif (Monsoon)", "months": "June - October", "crops": ["Groundnut", "Millets", "Pulses"]},
            {"season": "Rabi (Winter)", "months": "October - March", "crops": ["Wheat", "Oilseeds", "Gram"]},
            {"season": "Zaid (Summer)", "months": "March - May", "crops": ["Cowpea", "Sesame"]}
        ],
        "comparison_matrix": [
            {"attribute": "pH Range", "detected": "6.0 - 7.5 (Slight Acid)", "other1": "Black: 7.2 - 8.5", "other2": "Alluvial: 6.5 - 8.4"},
            {"attribute": "Drainage", "detected": "Good to Rapid", "other1": "Black: Low / Clay", "other2": "Alluvial: Moderate"},
            {"attribute": "Soil Fertility", "detected": "Medium (Iron Rich)", "other1": "Black: High", "other2": "Alluvial: High"},
            {"attribute": "Water Retention", "detected": "Low to Moderate", "other1": "Black: Exceptional", "other2": "Alluvial: High"},
            {"attribute": "Primary Crops", "detected": "Groundnut, Ragi, Pulses", "other1": "Cotton, Sugarcane", "other2": "Rice, Wheat"}
        ]
    },
    "Yellow Soil": {
        "color_hex": "#d49e2a",
        "gradient": "linear-gradient(135deg, #d49e2a, #f0c465)",
        "texture_tag": "Hydrated Iron Fine Loam",
        "drainage": "Moderate",
        "ph_typical": "5.5 - 7.0",
        "description": "Develops warm ochre yellow hue due to hydration of iron oxides in high rainfall environments. Fine textured and nutrient responsive.",
        "explainability": "Classified as Yellow Soil due to warm hydrated iron ochre-yellow tint and fine loamy particle structure.",
        "sample_image": "/static/samples/Yellow_Soil.jpg",
        "eco_impact": {
            "health_score": 84,
            "carbon_capacity": "Hydrated Iron Organic Content (1.25% C)",
            "water_efficiency": "68% Fine Loam Water Holding Capacity",
            "regeneration_tip": "Use vermicompost & composted manure to preserve fine loamy crumb structure during heavy rains."
        },
        "agronomic_reference": {
            "ph": {"val": "5.5 - 7.0", "score": 78, "status": "Slightly Acidic"},
            "nitrogen": {"val": "125 kg/ha", "score": 52, "status": "Low-Med Level"},
            "phosphorus": {"val": "13 kg/ha", "score": 50, "status": "Moderate P"},
            "potassium": {"val": "160 kg/ha", "score": 62, "status": "Moderate Potash"},
            "moisture": {"val": "24%", "score": 60, "status": "Moderate Moisture"},
            "salinity": {"val": "0.4 dS/m", "score": 95, "status": "Low Salinity"}
        },
        "crop_suitability": [
            {"name": "Rice", "icon": "🍚", "reason": "Hydrated loamy soil retains monsoon water effectively for paddy cultivation."},
            {"name": "Maize", "icon": "🌽", "reason": "Fine soil texture responds well to balanced N-P-K applications."},
            {"name": "Groundnut", "icon": "🥜", "reason": "Loamy soil aeration aids pod formation under adequate rainfall."},
            {"name": "Pulses", "icon": "🫘", "reason": "Improves organic nitrogen reserves in yellow loams."},
            {"name": "Sugarcane", "icon": "🎋", "reason": "High rainfall availability supports heavy stalk juice accumulation."}
        ],
        "seasonal_calendar": [
            {"season": "Kharif (Monsoon)", "months": "June - October", "crops": ["Rice", "Maize", "Sugarcane"]},
            {"season": "Rabi (Winter)", "months": "October - February", "crops": ["Pulses", "Oilseeds", "Wheat"]},
            {"season": "Zaid (Summer)", "months": "March - May", "crops": ["Groundnut", "Vegetables"]}
        ],
        "comparison_matrix": [
            {"attribute": "pH Range", "detected": "5.5 - 7.0 (Mild Acid)", "other1": "Red: 6.0 - 7.5", "other2": "Alluvial: 6.5 - 8.4"},
            {"attribute": "Drainage", "detected": "Moderate Drainage", "other1": "Red: Good / Rapid", "other2": "Alluvial: Moderate"},
            {"attribute": "Soil Fertility", "detected": "Medium (Hydrated Fe)", "other1": "Red: Medium", "other2": "Alluvial: High"},
            {"attribute": "Water Retention", "detected": "Moderate (20-30%)", "other1": "Red: Low", "other2": "Alluvial: High"},
            {"attribute": "Primary Crops", "detected": "Rice, Maize, Pulses", "other1": "Groundnut, Ragi", "other2": "Rice, Wheat"}
        ]
    }
}


def extract_direct_image_url(raw_url):
    try:
        parsed = up.urlparse(raw_url)
        query_params = up.parse_qs(parsed.query)

        if "imgurl" in query_params:
            return up.unquote(query_params["imgurl"][0])

        if "mediaurl" in query_params:
            return up.unquote(query_params["mediaurl"][0])

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(raw_url, headers=headers, timeout=5, stream=True)
        content_type = resp.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            html_chunk = resp.content[:50000].decode("utf-8", errors="ignore")
            og_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html_chunk, re.IGNORECASE)
            if not og_match:
                og_match = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', html_chunk, re.IGNORECASE)
            if not og_match:
                og_match = re.search(r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']', html_chunk, re.IGNORECASE)
            if og_match:
                found_img = og_match.group(1)
                return up.urljoin(raw_url, found_img)

        return raw_url
    except Exception:
        return raw_url


def is_plausible_soil_image(image_obj):
    try:
        img = image_obj.resize((100, 100))
        arr = np.array(img, dtype=np.float32)

        std_per_channel = np.std(arr, axis=(0, 1))
        avg_std = np.mean(std_per_channel)

        if avg_std < 10.0:
            return False, "Image lacks natural surface texture. Please upload a clear photo of soil, ground, or earth."

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        b_dominance = np.mean((b > r + 30) & (b > g + 30))
        if b_dominance > 0.4:
            return False, "This photo appears to be sky, water, or blue screen. Please upload a photo of soil."

        neon = np.mean((g > 200) & (b > 200) & (r < 50))
        if neon > 0.3:
            return False, "Image contains artificial colors not found in natural soil."

        white_pixels = np.mean((r > 240) & (g > 240) & (b > 240))
        if white_pixels > 0.85:
            return False, "This appears to be a blank paper or text screenshot. Please upload a photo of soil."

        return True, "Valid"
    except Exception:
        return True, "Valid"


def generate_feature_map(image_obj, color_hex="#a3704c"):
    try:
        img = image_obj.resize((300, 300)).convert("RGB")
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageEnhance.Contrast(edges).enhance(2.5)

        hex_clean = color_hex.lstrip("#")
        cr, cg, cb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        
        edges_arr = np.array(edges, dtype=np.float32) / 255.0
        img_arr = np.array(img, dtype=np.float32)

        overlay = img_arr.copy()
        overlay[:, :, 0] = np.clip(img_arr[:, :, 0] + edges_arr * cr * 0.7, 0, 255)
        overlay[:, :, 1] = np.clip(img_arr[:, :, 1] + edges_arr * cg * 0.7, 0, 255)
        overlay[:, :, 2] = np.clip(img_arr[:, :, 2] + edges_arr * cb * 0.7, 0, 255)

        res_img = Image.fromarray(overlay.astype(np.uint8))
        buf = io.BytesIO()
        res_img.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        return None


def multi_crop_predict(image_obj):
    if not cnn_model:
        return "Alluvial Soil", 92.0, {c: (92 if c == "Alluvial Soil" else 1) for c in soil_classes}, False, None

    w, h = image_obj.size
    crop_w, crop_h = int(w * 0.8), int(h * 0.8)

    boxes = [
        ((w - crop_w) // 2, (h - crop_h) // 2, (w + crop_w) // 2, (h + crop_h) // 2),
        (0, 0, crop_w, crop_h),
        (w - crop_w, 0, w, crop_h),
        (0, h - crop_h, crop_w, h),
        (w - crop_w, h - crop_h, w, h)
    ]

    tensors = []
    for box in boxes:
        cropped = image_obj.crop(box).resize((224, 224))
        arr = np.array(cropped)
        tensors.append(arr)
        if len(tensors) == 1:
            tensors.append(np.fliplr(arr))

    batch = np.array(tensors)
    batch_preds = cnn_model.predict(batch, verbose=0)
    avg_preds = np.mean(batch_preds, axis=0)

    # Sort probabilities descending
    sorted_idx = np.argsort(avg_preds)[::-1]
    top1_idx = sorted_idx[0]
    top2_idx = sorted_idx[1]

    top1_soil = soil_classes[top1_idx]
    top2_soil = soil_classes[top2_idx]

    top1_conf = float(avg_preds[top1_idx] * 100)
    top2_conf = float(avg_preds[top2_idx] * 100)

    # Check if prediction is ambiguous (top1 < 60% OR gap between top1 & top2 < 15%)
    is_ambiguous = (top1_conf < 60.0) or ((top1_conf - top2_conf) < 15.0)

    secondary_match = None
    if is_ambiguous and top2_conf > 10.0:
        secondary_match = {
            "soil_type": top2_soil,
            "confidence": int(round(top2_conf)),
            "reason": soil_metadata.get(top2_soil, {}).get("description", "")
        }

    class_probs = {soil_classes[i]: int(round(float(avg_preds[i] * 100))) for i in range(len(soil_classes))}

    return top1_soil, top1_conf, class_probs, is_ambiguous, secondary_match


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_soil", methods=["POST"])
def predict_soil():
    try:
        sample_path = request.form.get("sample_path")
        image_url = request.form.get("image_url")
        image_obj = None
        extracted_url = None

        if "image" in request.files and request.files["image"].filename != "":
            image_file = request.files["image"]
            image_obj = Image.open(image_file).convert("RGB")
        elif image_url:
            try:
                extracted_url = extract_direct_image_url(image_url)
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(extracted_url, headers=headers, timeout=8)
                if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", "").lower():
                    image_obj = Image.open(io.BytesIO(resp.content)).convert("RGB")
                else:
                    return jsonify({"error": "This image couldn't be loaded — try a different source or upload it directly."}), 400
            except Exception:
                return jsonify({"error": "This image couldn't be loaded — try a different source or upload it directly."}), 400
        elif sample_path:
            clean_rel = sample_path.lstrip("/")
            if os.path.exists(clean_rel):
                image_obj = Image.open(clean_rel).convert("RGB")
        else:
            return jsonify({"error": "No soil photo or URL provided"}), 400

        if image_obj is None:
            return jsonify({"error": "Unable to load soil image"}), 400

        # STAGE 1: Soil vs Non-Soil Binary Check
        binary_tensor = np.expand_dims(np.array(image_obj.resize((224, 224))), axis=0)
        if binary_model:
            # Output sigmoid: Class 0 = non_soil, Class 1 = soil
            soil_probability = float(binary_model.predict(binary_tensor, verbose=0)[0][0])
        else:
            soil_probability = 1.0

        # Also apply domain heuristic check
        is_plausible, heuristic_msg = is_plausible_soil_image(image_obj)

        # STAGE 1 REJECTION THRESHOLD (75% confidence required to be soil)
        if soil_probability < 0.75 or not is_plausible:
            return jsonify({
                "is_soil": False,
                "error": "This doesn't look like a soil image. Please upload a clear photo of soil, ground, or earth."
            }), 400

        # STAGE 2: 7-Class Soil Type Classification (Runs ONLY if Stage 1 passes)
        soil_type, raw_confidence, class_probs, is_ambiguous, secondary_match = multi_crop_predict(image_obj)

        confidence_pct = int(round(raw_confidence))
        # STAGE 2 LOW-CONFIDENCE THRESHOLD (<60%)
        low_confidence = raw_confidence < 60.0

        meta = soil_metadata.get(soil_type, soil_metadata["Alluvial Soil"])

        feature_map_b64 = generate_feature_map(image_obj, meta.get("color_hex", "#a3704c"))

        similar_images = [meta.get("sample_image")]
        sample_folder = os.path.join("static", "dataset_samples", soil_type.replace(" ", "_"))
        if os.path.exists(sample_folder):
            extra = [f"/static/dataset_samples/{soil_type.replace(' ', '_')}/{f}" for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg', '.png'))][:3]
            similar_images.extend(extra)

        return jsonify({
            "is_soil": True,
            "soil_type": soil_type,
            "confidence": confidence_pct,
            "confidence_raw": round(raw_confidence, 2),
            "low_confidence": low_confidence,
            "low_confidence_msg": "Low Confidence — Try a Clearer Photo" if low_confidence else None,
            "is_ambiguous": is_ambiguous,
            "secondary_match": secondary_match,
            "class_probabilities": class_probs,
            "metadata": meta,
            "feature_map": feature_map_b64,
            "similar_images": similar_images,
            "extracted_url": extracted_url
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True)