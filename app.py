from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("waste_classifier.h5")

# Class labels
labels = ['ewaste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'textile']

# Waste explanation dictionary
waste_info = {
    "glass": """<b>Meaning:</b><br>
        Glass waste includes bottles, jars, containers, and broken glass items made from silica-based materials. It is non-biodegradable but can be recycled indefinitely without losing its quality.<br><br>
        <b>How to handle it:</b><br>
        Glass should be rinsed and cleaned before recycling. It must be placed in separate glass recycling bins. Broken glass should be handled carefully and disposed of safely to avoid injury.<br><br>
        <b>If dumped in open:</b><br>
        Glass does not decompose and remains in the environment for thousands of years. Broken pieces can scatter and create hazards for humans and animals. It also contributes to land pollution.<br><br>
        <b>Health consequences:</b><br>
        Sharp glass fragments can cause serious cuts and injuries. Open wounds from glass can lead to infections if not treated properly. It also increases risk of accidents in public areas.<br><br>
        <b>Final note:</b><br>
        Glass contributes about 5–7% of global waste and is one of the most recyclable materials.<br><br>""",

    "metal": """<b>Meaning:</b><br>
Metal waste includes cans, tins, foils, and other metallic objects made from aluminum, steel, or other metals. These materials are durable and can be recycled multiple times.<br><br>
<b>How to handle it:</b><br>
Metal items should be cleaned before recycling. They should be sent to scrap yards or recycling centers where they can be melted and reused. Proper segregation helps improve recycling efficiency.<br><br>
<b>If dumped in open:</b><br>
Metal waste can rust over time and contaminate soil and water. It contributes to environmental pollution and wastes valuable resources that could be reused.<br><br>
<b>Health consequences:</b><br>
Sharp edges of metal waste can cause injuries. Some metals may release harmful substances over time, affecting both human health and ecosystems.<br><br>
<b>Final note:</b><br>
Metal waste makes up around 5–10% of global waste and is highly recyclable.<br><br>""",

    "organic": """<b>Meaning:</b><br>
Organic waste includes biodegradable materials such as food scraps, vegetable peels, fruit waste, and garden waste. These materials break down naturally through microbial activity. They are commonly produced in households, restaurants, and agricultural activities.<br><br>
<b>How to handle it:</b><br>
Organic waste should be composted using compost bins or pits. It can be converted into nutrient-rich fertilizer within a few weeks. It should always be separated from plastic, metal, and glass waste. Home composting is one of the best eco-friendly solutions.<br><br>
<b>If dumped in open:</b><br>
It decomposes and produces foul smell and methane gas, which is a greenhouse gas. It attracts insects, rodents, and stray animals, creating unhygienic surroundings. It also contributes to landfill overflow and environmental pollution.<br><br>
<b>Health consequences:</b><br>
Can lead to bacterial growth and spread of diseases. Attracts flies and mosquitoes, increasing the risk of infections like dengue and malaria. Poor waste handling may also cause respiratory discomfort due to foul odor and gases.<br><br>
<b>Final note:</b><br>
Organic waste makes up approximately 40–50% of global waste, making it the largest category worldwide.<br><br>""",

    "paper": """<b>Meaning:</b><br>
Paper waste includes newspapers, books, office paper, cardboard, and packaging materials made from wood pulp. It is biodegradable and widely used in daily life.<br><br>
<b>How to handle it:</b><br>
Paper should be kept clean and dry before recycling. It must be separated from wet or contaminated waste. Recycling paper helps in producing new paper products and reduces the need for cutting trees.<br><br>
<b>If dumped in open:</b><br>
Paper decomposes but contributes to landfill waste and produces methane gas in large quantities. It also increases environmental burden if not recycled properly.<br><br>
<b>Health consequences:</b><br>
Wet paper waste can attract pests and promote bacterial growth. It can create unhygienic conditions and increase risk of infections.<br><br>
<b>Final note:</b><br>
Paper waste contributes about 15–20% of global waste, making it one of the major waste categories.<br><br>""",

    "plastic": """<b>Meaning:</b><br>
Plastic waste includes items like bottles, containers, packaging materials, wrappers, and disposable products made from synthetic polymers. These materials are non-biodegradable and persist in the environment for hundreds of years.<br><br>
<b>How to handle it:</b><br>
Plastic should be cleaned, dried, and sorted based on type before recycling. It should be placed in designated recycling bins and not mixed with organic waste. Reusing plastic items whenever possible also helps reduce waste generation.<br><br>
<b>If dumped in open:</b><br>
Plastic accumulates in land and water bodies, causing severe environmental pollution. It blocks drainage systems and contributes to urban flooding. It also harms animals, especially marine life, when ingested.<br><br>
<b>Health consequences:</b><br>
Burning plastic releases toxic gases like dioxins that damage the lungs and can cause serious diseases. Microplastics can enter the food chain and affect human health over time. Long-term exposure can lead to hormonal and respiratory issues.<br><br>
<b>Final note:</b><br>
Plastic accounts for around 10–12% of global waste, but has one of the highest environmental impacts.<br><br>""",

    "textile": """<b>Meaning:</b><br>
Textile waste includes discarded clothes, fabrics, and other cloth-based materials. It includes both natural and synthetic fibers used in the fashion industry.<br><br>
<b>How to handle it:</b><br>
Textiles should be reused, donated, or repurposed whenever possible. Old clothes can be recycled into new fabric or used for other purposes like cleaning cloths. Avoid throwing them into general waste.<br><br>
<b>If dumped in open:</b><br>
Textile waste accumulates in landfills and takes years to decompose, especially synthetic fabrics. It releases microfibers into the environment, contributing to pollution.<br><br>
<b>Health consequences:</b><br>
Synthetic textiles can release harmful chemicals into soil and water. Long-term exposure to these pollutants can affect both environmental and human health.<br><br>
<b>Final note:</b><br>
Textile waste accounts for approximately 5% of global waste and is rapidly increasing due to fast fashion.<br><br>""",

    "ewaste": """<b>Meaning:</b><br>
E-waste includes electronic items such as chargers, cables, batteries, circuit boards, and other electrical devices. It contains both valuable and hazardous materials.<br><br>
<b>How to handle it:</b><br>
E-waste should be taken to certified recycling centers. It should never be disposed of in regular garbage. Proper recycling helps recover valuable metals and prevents environmental damage.<br><br>
<b>If dumped in open:</b><br>
It releases toxic substances like lead, mercury, and cadmium into soil and water. This causes long-term environmental pollution and damages ecosystems.<br><br>
<b>Health consequences:</b><br>
Exposure to e-waste toxins can cause serious health issues including neurological damage, respiratory problems, and organ damage. It is one of the most hazardous waste types.<br><br>
<b>Final note:</b><br>
E-waste makes up about 3–5% of global waste, but is one of the most dangerous categories.<br><br>"""
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_file = None
    info = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            if not os.path.exists("static"):
                os.makedirs("static")

            img_path = os.path.join("static", file.filename)
            file.save(img_path)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img)
            class_id = np.argmax(preds)
            prediction = labels[class_id]
            confidence = round(preds[0][class_id] * 100, 2)

            image_file = file.filename
            info = waste_info[prediction]  # ← THIS WAS MISSING

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_file=image_file,
                           info=info)


if __name__ == "__main__":
    app.run(debug=True)