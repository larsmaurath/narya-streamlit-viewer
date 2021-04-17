import streamlit as st
import pandas as pd
from helpers import get_table_download_link
from narya.narya.tracker.full_tracker import FootballTracker
import cv2
import numpy as np

from mplsoccer.pitch import Pitch

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

st.set_option('deprecation.showfileUploaderEncoding', False)

image = None

# This is a work-around to disable the entity tracker and analyze multiple frames
# With the same tracker
@st.cache(allow_output_mutation=True)
def create_players_to_remove():
    return []

@st.cache(allow_output_mutation=True)
def create_tracker():
    tracker = FootballTracker(pretrained=True, 
                              frame_rate=23,
                              track_buffer = 0,
                              ctx=None)
        
    return tracker

template = cv2.imread('narya/world_cup_template.png')
template = cv2.resize(template, (512,512))/255.

image_selection = st.selectbox('Choose image:', ['', 'Example Image', 'Upload Image'], 
                        format_func=lambda x: 'Choose image' if x == '' else x)

if image_selection:
    if image_selection == 'Example Image':
        image = cv2.imread("atm_che_23022021_62_07_2.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    else:
        
        st.title('Upload Image')
        uploaded_file = st.file_uploader("Select Image file to open:", type=["png", "jpg"])
        
        if uploaded_file:
                
            file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is not None:
    
        img_list = [image]
        
        tracker = create_tracker()
        players_to_remove = create_players_to_remove()
        
        
        try:
        
            trajectories = tracker(img_list,
                                   split_size = 512, 
                                   save_tracking_folder = 'narya_output/', 
                                   template = template, 
                                   skip_homo = [])
            
            
            trajectories = {k: v for k, v in trajectories.items() if k not in players_to_remove}
            
            players_to_remove += list(trajectories.keys())
            
            x_coords = [val[0][0] for val in trajectories.values()]
            y_coords = [val[0][1] for val in trajectories.values()]
            colors = [val[0][3] for val in trajectories.values()]
            colors = [tuple([item / 255 for item in subl]) for subl in colors]
        
            x_coords = [x/320*120 for x in x_coords]
            y_coords = [y/320*80 for y in y_coords]
        
            plot = Pitch(view='full', figsize=(6.8, 10.5), orientation='horizontal')
            fig, ax = plot.draw()
            
            plot.scatter(x_coords, y_coords, ax=ax, c='#c34c45', s=150)
        
            st.title('Tracking Results')
        
            st.write('From left to right: the original image, overlayed bounding boxes + homography and a schematic represenation',
                                              expanded=True)
            
            col1, col2, col3 = st.beta_columns(3)
            
            with col1:
                st.image(image, use_column_width= 'always')
    
            with col2:
                image_processed = cv2.imread("narya_output/test_00000.jpg")
                image_processed = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)
                st.image(image_processed, use_column_width= 'always')
    
            with col3: 
                
                st.pyplot(fig)
            
            review = st.selectbox('Do the results look good?:', ['', 'Yes and export', 'No and manually fix'], 
                                    format_func=lambda x: 'Do the results look good?' if x == '' else x)
            
            if review:
                if review == 'Yes and export':
                    df = pd.DataFrame({'x': x_coords, 'y': y_coords})
                    st.markdown(get_table_download_link(df[['x', 'y']]), unsafe_allow_html=True)
                    
                else:
                    st.write("I hope to soon add the functionality to annotate players and homography manually")
                    
        except:
            
            st.write('The model has not found enough keypoints')