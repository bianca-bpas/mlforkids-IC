import io
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

import numpy as np
import urllib.request, urllib.error, json
from time import sleep

#
# Helper class for training an image classifier using training data
#  from the Machine Learning for Kids website.
#
class MLforKidsImageProject:

    IMAGESIZE=(224,224)
    INPUTLAYERSIZE=IMAGESIZE + (3,)

    # scratchkey is the secret API key that allows access to training
    #  data from a single project on the MLforKids website
    def __init__(self, scratchkey: str):
        # register custom HTTP handler
        opener = urllib.request.build_opener(MLforKidsHTTP())
        urllib.request.install_opener(opener)

        print("MLFORKIDS: Baixando informações sobre o projeto de aprendizado de máquina")
        self.scratchkey = scratchkey
        try:
            apiurl = self.__switchToTemporarySite("https://machinelearningforkids.co.uk/api/scratch/" + scratchkey + "/train")
            with urllib.request.urlopen(apiurl) as url:
                self.__downloaded_training_images_list = json.loads(url.read().decode())
        except urllib.error.HTTPError:
            raise RuntimeError("Unable to retrieve machine learning project - please check that the key is correct")

    # Generates a name for the local cache file where the downloaded training
    #  image is saved. An image file extension is required, otherwise it will
    #  be ignored by ImageDataGenerator.
    def __get_fname(self, trainingitem):
        extension = ".png" if trainingitem["imageurl"].lower().endswith(".png") else ".jpg"
        return trainingitem["id"] + extension

    # Downloads all of the training images for this project, and sets up an
    #  ImageDataGenerator against the folder where they have been downloaded
    def __get_training_images_generator(self):
        print("MLFORKIDS: Baixando imagens de exemplos no mlForKids para treinar o modelo... (isso vai demorar um pouco)\n\n")
        cachedir = "~/.keras/"
        cachelocation = os.path.join("datasets", "mlforkids", self.scratchkey)
        projectcachedir = str(os.path.expanduser(os.path.join(cachedir, cachelocation)))

        # ('id', 'b5384150-9488-11ee-ba06-71d8cd6f1d9a')
        # ('imageurl', 'https://www.cin.ufpe.br/~wjls/projeto%20IC/SINAL-LIBERADO/00049.jpg')
        # ('label', 'sinal_liberado')
        for trainingitem in self.__downloaded_training_images_list:
            try:
                original_stdout = sys.stdout
                sys.stdout = io.StringIO()
                tf.keras.utils.get_file(origin=self.__switchToTemporarySite(trainingitem["imageurl"]),
                                        cache_dir=cachedir,
                                        cache_subdir=os.path.join(cachelocation, trainingitem["label"]),
                                        fname=self.__get_fname(trainingitem))
                sys.stdout = original_stdout

                print(f"{trainingitem['imageurl'].split('/')[-1]} : imagem baixada! - {trainingitem['label']}")
                # avoid common rate-limiting errors by pausing
                #  for a quarter-second between each download
                sleep(0.25)
            except Exception as downloaderr:
                print("ERROR: Unable to download training image from", trainingitem["imageurl"])
                print(downloaderr)
                print("ERROR: Skipping training image and continuing without it", trainingitem["imageurl"])

        return ImageDataGenerator().flow_from_directory(str(projectcachedir),
                                                        target_size=MLforKidsImageProject.IMAGESIZE)

    # Creates a lookup table for the classes that this project is being trained
    #  to recognize.
    # TODO : dumb implementation - should rewrite
    def __get_class_lookup(self, training_image_data):
        class_labels = [None]*training_image_data.num_classes
        class_names = training_image_data.class_indices.keys()
        for classname in class_names:
            class_labels[training_image_data.class_indices[classname]] = classname
        return class_labels

    # Defines a simple image classifier based on a mobilenet model from TensorFlow hub
    def __define_model(self):
        print("\n\nMLFORKIDS: Definindo camadas a serem incluídas na rede neural")
        model = Sequential([
            # input layer is resizing all images to save having to do that in a manual pre-processing step
            Rescaling(1/127, input_shape=MLforKidsImageProject.INPUTLAYERSIZE),
            # using an existing pre-trained model as an untrainable main layer
            hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5"),
            #
            Dropout(rate=0.2),
            #
            Dense(self.num_classes)
        ])
        model.build((None,) + MLforKidsImageProject.INPUTLAYERSIZE)

        # model compile parameters copied from tutorial at https://www.tensorflow.org/hub/tutorials/tf2_image_retraining
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    # Runs the model fit function to train the tl model
    def __train_model(self, trainingimagesdata):
        print("MLFORKIDS: Iniciando treinamento do modelo... (isso também vai demorar)\n\n")
        if trainingimagesdata.batch_size > trainingimagesdata.samples:
            trainingimagesdata.batch_size = trainingimagesdata.samples
        steps_per_epoch = trainingimagesdata.samples // trainingimagesdata.batch_size
        epochs = 8
        if trainingimagesdata.samples > 55:
            epochs = 15
        self.ml_model.fit(trainingimagesdata, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=0)
        print("MLFORKIDS: Treinamento completo!\n\n")


    # Cloudflare is currently blocking access to the Machine Learning for Kids API
    #  from non-browser user agents
    # While I raise this with them to get this unblocked, switching to this
    #  temporary URL should avoid the problem
    #
    # TODO: remove this function as soon as Cloudflare have
    #  stopped breaking Python apps
    #
    def __switchToTemporarySite(self, url):
        return url.replace("https://machinelearningforkids.co.uk/api/scratch/",
                           "https://mlforkids-api.j8clybxvjr0.us-south.codeengine.appdomain.cloud/api/scratch/")


    #
    # public methods
    #

    # Fetches the training data for this project, and uses it to train a machine learning model
    def train_model(self):
        training_images = self.__get_training_images_generator()
        self.num_classes = training_images.num_classes
        self.ml_class_names = self.__get_class_lookup(training_images)
        self.ml_model = self.__define_model()
        self.__train_model(training_images)

    # Returns a prediction for the image at the specified location
    def prediction(self, image_location: str):
        if hasattr(self, "ml_model") == False:
            raise RuntimeError("Machine learning model has not been trained for this project")
        testimg = image.load_img(image_location, target_size=MLforKidsImageProject.IMAGESIZE)
        testimg = image.img_to_array(testimg)
        testimg = np.expand_dims(testimg, axis=0)
        predictions = self.ml_model.predict(testimg)
        topprediction = predictions[0]
        topanswer = np.argmax(topprediction)
        return {
            "class_name": self.ml_class_names[topanswer],
            "confidence": 100 * np.max(tf.nn.softmax(topprediction))
        }


#
# Helper class for making HTTP requests to fetch training images
#  for machine learning projects
#
# It adds a user-agent header so that when scraping images from
#  third-party websites, the Python code correctly identifies
#  itself, so that appropriate rate-limiting can be applied.
#
class MLforKidsHTTP(urllib.request.HTTPHandler):
    def http_request(self, req):
        req.headers["User-Agent"] = "MachineLearningForKidsPythonBot/1.0"
        return super().http_request(req)






# from mlforkids import MLforKidsImageProject
from google.colab import files
import os


def analisarImg(filename):
    # get image
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    demo = myproject.prediction(filename)
    sys.stdout = original_stdout

    label = demo["class_name"]
    confidence = int(demo["confidence"])

    if label=="green":
        label = "LIBERADO"
    else:
        label = "NÃO LIBERADO"

    return [label, confidence]

def uploadImg():
    uploaded = files.upload()

    for filename in uploaded.keys():
    	return filename

def test():
    testImages = os.listdir("/content/test_images")

    points = 0
    print("TESTE: analisando imagens do diretório test_images")

    for img in testImages:
        imgName = img.split("-")[0] + ".jpg"
        situation = img.split("-")[1].split(".")[0]

        if situation=="green":
            label = "LIBERADO"
        else:
            label = "NÃO LIBERADO"

        result = analisarImg("/content/test_images/" + img)
        if label==result[0]:
            print(f"{imgName} : Esperado - {label} ; Obtido - {result[0]} ({result[1]}%) ; SUCESSO!")
            points+=1
        else:
            print(f"{imgName} : Esperado - {label} ; Obtido - {result[0]} ({result[1]}%) ; FALHA!")

    print(f"\n\nAnalisando as {len(testImages)} imagens enviadas, a precisão final do algoritmo foi de {int(points/len(testImages)*100)}%")


# mlforkids key
key = "c6740a40-937e-11ee-98c7-c79c85a0194ea582b4a0-1349-40ec-b4d9-aca12f697527"

print("Semáforo Livre ou Não?\n\n")

# training model
myproject = MLforKidsImageProject(key)
myproject.train_model()

print("\nEnvie uma imagem de sinal de trânsito e descubra se ele está liberado ou não.")


while True:
    print("\n1. Enviar imagem.")
    print("2. Teste")
    print("3. Sair\n")

    respMenu = input()
    if respMenu=="1":
        filenameUploaded = uploadImg()
        result = analisarImg(filenameUploaded)
        print(f"\n{filenameUploaded} : Semáforo {result[0]} - {result[1]}% de confiança!")
    elif respMenu=="2":
        test()
    else:
        break
