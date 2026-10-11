import io
import base64
from flask import Flask, request, render_template_string
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
model = YOLO("best.pt")

PAGE = """
<!doctype html>
<title>License Plate Detection</title>
<body style="font-family: Arial; max-width: 700px; margin: 40px auto; text-align: center;">
  <h2>License Plate Detection</h2>
  <p>Upload a car photo and the model will find the plate.</p>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Detect</button>
  </form>
  {% if result_img %}
    <h3>Plates found: {{ count }}</h3>
    <img src="data:image/jpeg;base64,{{ result_img }}" style="max-width:100%;">
  {% endif %}
</body>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result_img = None
    count = 0
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        result = model.predict(img, conf=0.4, imgsz=640)[0]
        count = len(result.boxes)
        plotted = result.plot()[..., ::-1]
        buf = io.BytesIO()
        Image.fromarray(plotted).save(buf, format="JPEG")
        result_img = base64.b64encode(buf.getvalue()).decode()
    return render_template_string(PAGE, result_img=result_img, count=count)

if __name__ == "__main__":
    app.run(debug=True)