import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from Working import Generator,GenNet,device
st.set_page_config(layout="wide", page_title="Anime Characters Generator")
st.image('animeimage.png')
num_images=st.number_input("Pick  the no. of characters you want to generate",1,1000)
if st.button("Generate"):
    def generate_anime_faces(num_images):
        GenNet.eval()
        with torch.no_grad():
            noise = torch.randn(num_images, 100, 1, 1, device=device)
        generated_images = GenNet(noise).cpu()

        plt.figure(figsize=(8, 8))
        for i in range(num_images):
            plt.subplot(4, 4, i+1)
            img = generated_images[i] / 2 + 0.5
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.axis('off')
        plt.show()
generate_anime_faces(num_images)