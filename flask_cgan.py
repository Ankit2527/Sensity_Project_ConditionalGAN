from flask import Flask, request, render_template, Response
import torch
from torch.autograd import Variable
from network  import Generator
import io  
import os
import argparse
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Set the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the noise and the image dimensions
noise_dim = 64
img_dim = 784

# Load generator model and move to desired device
netG = Generator(img_dim = img_dim, noise_dim = noise_dim).to(device)

# Load model
netG.load_state_dict(torch.load('./Sensity_Project_Results/CGAN/netG.pth'))

# Instiate the Flask app
app = Flask(__name__, template_folder= 'template')

@app.route('/')
def index():
    return render_template('file.html')

@app.route('/', methods=['POST']) 
def display_image():
    digit = request.form['u']
    img = generate_figure(digit)
    output = io.BytesIO()
    FigureCanvas(img).print_png(output)
    return Response(output.getvalue(), mimetype = 'image/png')

def generate_figure(digit):
    with torch.no_grad():
        f_noise = Variable(torch.randn(1, noise_dim)).to(device)
        f_label = torch.LongTensor([int(digit)]).to(device)
        f_image = netG(f_noise, f_label, noise_dim).cpu()
        f_image = make_grid(f_image, nrow = 1, normalize = True)
        _, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(f_image.permute(1, 2, 0).data, cmap = 'gray')
        fig = ax.figure
        #path = './Sensity_Project_Results/'
        #fname = 'class_image' + '.png'
        #fname = os.path.join(path, fname)
        #fig.savefig(fname, bbox_inches = 'tight')
        return fig

# Define the route
if __name__ == '__main__':
    app.run(debug = True)

# def flask_generate(device, noise_dim, img_dim):
#     """
#     Returns a Flask API locat host to generate an image for a corresponding class
#     Inputs:
#         device - Device to use for training.
#         noise_dim - Dimension of the noise vector.
#         img_dim - Dimension of the images(28 * 28).
#     """

#     # Load generator model and move to desired device
#     netG = Generator(img_dim = img_dim, noise_dim = noise_dim).to(device)

#     # Load model
#     netG.load_state_dict(torch.load('./Sensity_Project_Results/CGAN/netG.pth'))

#     # Instiate the Flask app
#     app = Flask(__name__, template_folder= 'template')

#     @app.route('/')
#     def index():
#         return render_template('file.html')

#     @app.route('/', methods=['POST']) 
#     def display_image():
#         digit = request.form['u']
#         img = generate_figure(digit)
#         output = io.BytesIO()
#         FigureCanvas(img).print_png(output)
#         return Response(output.getvalue(), mimetype = 'image/png')

#     def generate_figure(digit):
#         with torch.no_grad():
#             f_noise = Variable(torch.randn(1, noise_dim)).to(device)
#             f_label = torch.LongTensor([int(digit)]).to(device)
#             f_image = netG(f_noise, f_label, noise_dim).cpu()
#             f_image = make_grid(f_image, nrow = 1, normalize = True)
#             _, ax = plt.subplots(figsize=(8, 4))
#             ax.imshow(f_image.permute(1, 2, 0).data, cmap = 'gray')
#             fig = ax.figure
#             #path = './Sensity_Project_Results/'
#             #fname = 'class_image' + '.png'
#             #fname = os.path.join(path, fname)
#             #fig.savefig(fname, bbox_inches = 'tight')
#             return fig

#     # Define the route
#     if __name__ == '__main__':
#         app.run(debug = True)

# if __name__ == '__main__':

#     # Command line arguments
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--img_dim', default = 28 * 28, type = int,
#                         help = 'Size of FashionMNIST images which is 784')  
#     parser.add_argument('--noise_dim', default = 64, type = int,
#                         help = 'Dimension of the noise vector')      

#     args = parser.parse_args()
#     # Use GPU if available, else use CPU
#     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

#     kwargs = vars(args)
#     flask_generate(**kwargs)