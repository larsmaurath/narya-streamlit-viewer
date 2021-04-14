import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from helpers import Homography, VoronoiPitch, Play, PitchImage, PitchDraw, get_table_download_link
from pitch import FootballPitch
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

@st.cache(allow_output_mutation=True)
def create_tracker():
    tracker = FootballTracker(pretrained=True, 
                              frame_rate=23,
                              track_buffer = 60,
                              ctx=None)
        
    return tracker

template = cv2.imread('narya/world_cup_template.png')
template = cv2.resize(template, (512,512))/255.

image_selection = st.selectbox('Choose image:', ['', 'Example Image', 'My own image'], 
                        format_func=lambda x: 'Choose image' if x == '' else x)

if image_selection:
    if image_selection == 'Example Image':
        image = cv2.imread("atm_che_23022021_62_07_2.jpg")
        
    else:
        
        st.title('Upload Image or Video')
        uploaded_file = st.file_uploader("Select Image file to open:", type=["png", "jpg", "mp4"])
        pitch = FootballPitch()
        
        if uploaded_file:
            if uploaded_file.type == 'video/mp4':
                play = Play(uploaded_file)
                t = st.slider('You have uploaded a video. Choose the frame you want to process:', 0.0,60.0)
                image = play.get_frame(t)
                image = cv2.imread("atm_che_23022021_62_07_2.jpg")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            else:
                
                file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is not None:
    
        img_list = [image]
        
        tracker = create_tracker()
        
        trajectories = tracker(img_list,
                               split_size = 512, 
                               save_tracking_folder = 'narya_output/', 
                               template = template, 
                               skip_homo = [])
        
        x_coords = [val[0][0] for val in trajectories.values()]
        y_coords = [val[0][1] for val in trajectories.values()]
    
        x_coords = [x/320*120 for x in x_coords]
        y_coords = [y/320*80 for y in y_coords]
    
        pitch = Pitch(view='full', figsize=(6.8, 10.5), orientation='horizontal')
        fig, ax = pitch.draw()
        
        pitch.scatter(x_coords, y_coords, ax=ax, c='#c34c45', s=150)
    
        st.title('Tracking Results')
    
        st.write('From left to right: the original image, overlayed bounding boxes + homography and a schematic represenation',
                                          expanded=True)
        
        col1, col2, col3 = st.beta_columns(3)
        
        with col1:
            st.image(image, use_column_width= 'always')

        with col2:
            st.image("narya_output/test_00000.jpg", use_column_width= 'always')

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