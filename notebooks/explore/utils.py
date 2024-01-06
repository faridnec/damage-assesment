import io
import ipywidgets as widgets

from ipywidgets import HBox, Output, Image as IPImage
import folium 
from folium.plugins import FastMarkerCluster
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np 
import pandas as pd

from IPython.display import clear_output, display
from ipyfilechooser import FileChooser

from typing import List, Dict, Tuple, Any, Callable


def images_on_server(display: display) -> Dict['str', Any]:
    ''' Display a file chooser widget that allows to select files that are in the server side

    Args:
        display (display): The widget display to render the ouputs
    Returns:
        _ (Dict): Dictionary that returns the filechooser and main display
    '''
    # fc = FileChooser('../../data/test')
    
    # Get the absolute path to the project directory
    project_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))

    # Construct the absolute path to the data/test/ directory
    data_test_path = os.path.join(project_dir, 'data', 'test')

    # Use the absolute path in the FileChooser
    fc = FileChooser(data_test_path)
    
    main_display1 = widgets.Output(layout={'height': '400px', 'width': '30%'})
    main_display2 = widgets.Output(layout={'height': '400px', 'width': '70%'})
    main_display = HBox([main_display1, main_display2])
    
    def show_example(chooser):
        filename = chooser.selected
        tokens = filename.split("\\") #
        loc, no, lon, lat = ("", "", "", "")  # Default values
        split_result = tokens[-1].replace(".png", "").rsplit("_", 1)
        if len(split_result) == 2:
            loc, no, lon, lat = tuple(split_result[0].rsplit("_", 2)) + (split_result[1],)
        
        with  main_display1:
            main_display1.clear_output()
            print(f"set: { tokens[-3]}")
            print(f"class: { tokens[-2]}")
            print(f'loc: {loc}')
            print(f'no: {no}')
            print(f"lat: {lat}")
            print(f"lon: {lon}")
            image = Image.open(chooser.selected)
            # display(image)
            
            # Resize the image (change the dimensions as needed)
            resized_image = image.resize((300, 300))
            display(resized_image)
                         
        with  main_display2:
            main_display2.clear_output()
            mapit = folium.Map(width=500,height=500,location=[lat, lon], zoom_start=7 )
            folium.Marker( location=[lat, lon], fill_color='#43d9de', radius=8 ).add_to( mapit )
            display(mapit)

    fc.register_callback(show_example)
    
    return {'fileChooser': fc, 'output': main_display}

def get_dataframe_from_file_structure() -> pd.core.frame.DataFrame:
    ''' Creates a dataframe with metadata based on the file structure.
    
    Returns:
        _ (pd.core.frame.DataFrame): Dataframe with metadata
    '''
    # Dataset paths
    base = os.path.abspath(os.path.join(os.getcwd(), '../../data'))
    subsets = ['train', 'validation', 'test']
    labels = ['visible_damage', 'no_damage']

    # Navigate through every folder and its contents to create a dataframe
    data = []
    for seti in subsets:
        for label in labels:
            files = os.listdir(os.path.join(base, seti, label))
            for filename in files:
                path = os.path.join(seti, label, filename)
                loc, no, lon, lat = filename.replace(".png", "").split("_")
                data.append([seti, label, loc, no, lat, lon, path, filename])

    # Create dataframe
    return pd.DataFrame(data = data, columns=['subset', 'label', 'loc','no','lon', 'lat', 'path', 'filename'])

def interactive_plot_pair(base: str, matches: List[str]) -> Callable:
    '''Create a plot to visualize a pair of images at the same location. One
    showing damage and the other showing no damage.
    
    Args:
        base (str): The base of the image path
        matches (List[str]): a list of image names
    Returns:
        plot_image_pairs (Callable): A function that plots the image pairs given the image index
    '''
    def plot_pairs(base, matches, index):
        fig = plt.figure(figsize=(12, 12))
        ax = []

        im_damage = Image.open(os.path.join(base, 'visible_damage', matches[index])).resize((200, 200))
        im_nodamage = Image.open(os.path.join(base, 'no_damage', matches[index])).resize((200, 200))

        ax.append(fig.add_subplot(1, 2, 1))
        ax[-1].set_title("No damage") 
        ax[-1].axis('off')
        plt.imshow(im_nodamage)

        ax.append(fig.add_subplot(1, 2, 2))
        ax[-1].set_title("Visible damage") 
        ax[-1].axis('off')
        plt.imshow(im_damage)
        plt.axis('off')
        plt.show()


    def plot_image_pairs(file_index):
        plot_pairs(base, matches, index=file_index)
        
    return plot_image_pairs