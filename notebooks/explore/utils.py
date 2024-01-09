import io
import base64
import ipywidgets as widgets
import tempfile

from ipywidgets import HBox, Output, Image as IPImage
import folium
from folium import IFrame
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
    
    main_display1 = widgets.Output(layout={'height': '400px', 'width': '40%'})
    main_display2 = widgets.Output(layout={'height': '350px', 'width': '60%'})
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
            folium.Marker( location=[lat, lon], fill_color='#43d9de', radius=10 ).add_to( mapit )
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
    # subsets = ['test']
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

def image_to_base64(image_path):
    '''
    Convert an image file to a Base64-encoded string.

    This function reads the content of the image file specified by 'image_path',
    base64 encodes the binary data, and returns the resulting string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The Base64-encoded string representing the image.
    '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def resize_image(image_path, output_size):
    image = Image.open(image_path)
    resized_image = image.resize(output_size, Image.ANTIALIAS)
    return resized_image

def leaflet_plot(n_samples: int=200) -> folium.Map:
    '''
    Create a plot to visualize two sets of geo points: the ones with satellite
    images of damage and ones with satellite images of no damage done by hurricane.
    
    Args:
        n_samples (int): Number of points to show on the plot for each class
    Returns:
        map3 (folium.Map): The map with the points
    '''
    
    def icon_creator(size):
        return """
        function(cluster) {
          var childCount = cluster.getChildCount(); 
          var c = ' marker-cluster-';
          return new L.DivIcon({ html: '<div><span>' + childCount + '</span></div>', 
                                 className: 'marker-cluster'+c, 
                                 iconSize: new L.Point(40, 40) });
        }
        """.replace('marker-cluster-', f'marker-cluster-{size}')
    
    icon_create_function1 = icon_creator("large")
    icon_create_function2 = icon_creator("small")
    
    # Load datasets with n_samples of each label
    test_damage = load_coordinates('../../data/test/visible_damage', n_samples)
    test_nodamage = load_coordinates('../../data/test/no_damage', n_samples)
    
    map3 = folium.Map(location=[test_damage[0][1][0], test_damage[0][1][1]], tiles='CartoDB positron', zoom_start=10)

    marker_cluster_nodamage = FastMarkerCluster([], icon_create_function=icon_create_function2, min_child_weight=1, radius=10, maxClusterRadius=30).add_to(map3)
    
    for filename, point in test_nodamage:
        image_path = f'../../data/test/no_damage/{filename}'
        
        # Resize the image to a specific size
        resized_image = resize_image(image_path, output_size=(300, 300))

        # Save the resized image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file_path = temp_file.name
        resized_image.save(temp_file_path)
        temp_file.close()

        # Convert the resized image to Base64 using the temporary file path
        encoded_image = image_to_base64(temp_file_path)
        
        popup_text = (
        "<div style='font-family: Arial, sans-serif; color: green; text-align: center;'>"
        f"<b>No damage</b><br><img src='data:image/png;base64,{encoded_image}' width='200'>"
        "</div>"
        )
        
        iframe = IFrame(html=popup_text, width=250, height=250)
        popup = folium.Popup(iframe, max_width=300)
        
        folium.Marker(point, popup=popup, icon=folium.Icon(color="green")).add_to(marker_cluster_nodamage)

    marker_cluster_damage = FastMarkerCluster([], icon_create_function=icon_create_function1, min_child_weight=1, radius=10, maxClusterRadius=30).add_to(map3)
    
    for filename, point in test_damage:
        image_path = f'../../data/test/visible_damage/{filename}'
        
        # Resize the image to a specific size
        resized_image = resize_image(image_path, output_size=(300, 300))

        # Save the resized image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file_path = temp_file.name
        resized_image.save(temp_file_path)
        temp_file.close()

        # Convert the resized image to Base64 using the temporary file path
        encoded_image = image_to_base64(temp_file_path)
        
        popup_text = (
        "<div style='font-family: Arial, sans-serif; color: red; text-align: center;'>"
        f"<b>Visible damage</b><br><img src='data:image/png;base64,{encoded_image}' width='200'>"
        "</div>"
        )
        
        iframe = IFrame(html=popup_text, width=250, height=250)
        popup = folium.Popup(iframe, max_width=300)
        
        folium.Marker(point, popup=popup, icon=folium.Icon(color="red")).add_to(marker_cluster_damage)

    return map3

def load_coordinates(path: str, samples: int) -> List[Tuple[str, Tuple[float, float]]]:
    '''Load the  GPS coordinates from the first few samples in a given folder
    
    Args:
        path (str): path to the images
        samples (int): number of samples to take
    
    Returns:
        coordinates: An array containing the GPS coordianates extracted from the filenames
    '''
    files = os.listdir(path)
    coordinates = []
    indexes = list(range(len(files)))
    np.random.shuffle(indexes)
    indexes = indexes[0:samples]
    for i in indexes:
        # Get the coordinates
        parts = files[i].replace('.png', '').split('_')
        if len(parts) == 4:
            coordinate = (float(parts[3]), float(parts[2]))
            coordinates.append((files[i], coordinate))
        else:
            # Handle cases where the filename structure is different
            print(f"Skipping invalid filename: {files[i]}")
        
    return coordinates