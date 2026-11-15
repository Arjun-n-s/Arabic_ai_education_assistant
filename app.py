
from flask import Flask,render_template,request
import pickle
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from deep_translator import MyMemoryTranslator
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer,util

app=Flask(__name__)

model=load_model("final_caption_model.keras")
tokenizer=pickle.load(open("tokenizer.pkl","rb"))
max_length=37

resnet=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
encoder=Model(resnet.input,Reshape((49,2048))(resnet.output))

evaluation_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Both AI models loaded!")

def extract_features(image):
    image=Image.open(image).convert("RGB")
    image=image.resize((224,224))
    image=np.array(image,dtype="float32")
    image=np.expand_dims(image,axis=0)
    image=preprocess_input(image)
    return encoder.predict(image,verbose=0)

def generate_caption(feature):
    sequence=[tokenizer.word_index["startseq"]]
    for i in range(max_length-1):
        padded=pad_sequences([sequence],maxlen=max_length,padding="post")
        prediction=model.predict([feature,padded],verbose=0)
        word_id=int(np.argmax(prediction[0,len(sequence)-1]))
        word=tokenizer.index_word.get(word_id,"")
        if word in ["endseq",""]:
            break
        sequence.append(word_id)
    return " ".join(tokenizer.index_word.get(i,"") for i in sequence[1:])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/caption",methods=["GET","POST"])
def caption():
    if request.method=="POST":
        image=request.files["image"]
        image_bytes=image.read()

        feature=extract_features(BytesIO(image_bytes))
        english=generate_caption(feature)

        try:
            arabic=MyMemoryTranslator(source="en-GB",target="ar-SA").translate(english)
        except Exception:
            arabic="Translation unavailable"

        image_data=base64.b64encode(image_bytes).decode("utf-8")
        image_type=Image.open(BytesIO(image_bytes)).get_format_mimetype()

        return render_template("caption.html",english=english,arabic=arabic,image_data=image_data,image_type=image_type)

    return render_template("caption.html")

@app.route("/evaluation",methods=["GET","POST"])
def evaluation():
    if request.method=="POST":
        question=request.form["question"]
        reference=request.form["reference"]
        return render_template("student.html",question=question,reference=reference)
    return render_template("evaluation.html")

@app.route("/evaluate",methods=["POST"])
def evaluate():
    question=request.form["question"]
    reference=request.form["reference"]
    answer=request.form["answer"]
    embeddings=evaluation_model.encode([reference,answer],convert_to_tensor=True)
    similarity=util.cos_sim(embeddings[0],embeddings[1]).item()
    score=round(max(0,min(10,(similarity-0.3)/0.7*10)),2)
    return render_template("student.html",question=question,reference=reference,answer=answer,score=score)

if __name__=="__main__":
    app.run(debug=True)
