import pandas as pd
import xml.etree.ElementTree as ET
import os
import pathlib

def remove_xmlns(string, xmlns):
    """Removes a specified xmlns url from a string."""
    output = string.replace('{' + xmlns + '}', '')
    return output

def find(name, path):
    """Returns path of file if found in a directory."""
    for root, dirs, files in os.walk(path):
        if name in files:
            return pathlib.Path(root, name)

def is_panchromatic_product(xml_node):
    """Checks whether a satellite product is panchromatic."""
    for child in xml_node:
        if child.text == 'Pan':
            return True
    return False

def xml_root_children_to_dict(xml_root_element, xmlns):
    """Iterates through children of xml root. Returns dictionary."""
    d = {}
    for child in xml_root_element:
        d[remove_xmlns(child.tag, xmlns)] = child.text
    return d

def xml_product_to_dict(xml_product_element, xmlns):
    """Parses the <product> element of metadata xml file into dictionary."""
    d = {}
    count_strip, count_product, count_band = 0, 0, 0
    for prod_child in xml_product_element:
        prod_child_tag = remove_xmlns(prod_child.tag, xmlns)
        if prod_child_tag == 'strip':
            d[prod_child_tag+str(count_strip)] = xml_root_children_to_dict(prod_child, xmlns)
            count_strip += 1
        elif prod_child_tag == 'productFile':
            d[prod_child_tag+str(count_product)] = xml_root_children_to_dict(prod_child, xmlns)
            count_product += 1
        elif prod_child_tag == 'band':
            d[prod_child_tag+str(count_band)] = prod_child.text
            count_band += 1
        else:
            d[prod_child_tag] = prod_child.text
    d['n_bands'] = count_band
    d['n_products'] = count_product
    d['n_strips'] = count_strip
    return d

def xml_metadata_to_dict(path, xmlns):
    """Parses a metadata xml file into two dictionaries, a panchromatic and a multispectral."""
    d_pan, d_ms, d_mutual = {}, {}, {}
    xml_parsed = ET.parse(path)
    for child in xml_parsed.getroot():
        child_tag = remove_xmlns(child.tag, xmlns)
        if child_tag == 'product':
            if is_panchromatic_product(child):
                d_pan = xml_product_to_dict(child, xmlns)
            else: 
                d_ms = xml_product_to_dict(child, xmlns)
        else:
            d_mutual[child_tag] = child.text
    d_pan.update(d_mutual), d_ms.update(d_mutual)
    return d_pan, d_ms

def get_tif_path(metadata_dict, metadata_path, pan_or_ms):
    if pan_or_ms == 'pan':
        productfile = 'productFile29'
    elif pan_or_ms == 'ms':
        productfile = 'productFile28'
    else:
        raise ValueError('pan_or_ms argument must be eiter "pan" or "ms"')
    metadata_dir_path = metadata_path.parent
    tif_path = metadata_dict[productfile]['relativeDirectory']
    tif_filename = metadata_dict[productfile]['filename']
    return pathlib.Path(metadata_dir_path, tif_path, tif_filename)

def img_metadata_to_dict(metadata_name, data_path, xmlns, path_is_relative = True):
    """Parses all metadata xml files into a dictionary of dictionaries. Returns pan and ms dict."""
    data_path = pathlib.Path(data_path)
    if path_is_relative:
        data_path = pathlib.Path(pathlib.Path.cwd(), data_path) # Make absolute
    img_metadata_pan, img_metadata_ms = {}, {}
    img_list = os.listdir(data_path)
    for img in img_list:
        metadata_path = find(metadata_name, data_path.joinpath(img))
        img_metadata_pan[img], img_metadata_ms[img] = xml_metadata_to_dict(metadata_path, xmlns)
        img_metadata_pan[img]['tif_path'] = get_tif_path(img_metadata_pan[img], metadata_path, 'pan')
        img_metadata_ms[img]['tif_path'] = get_tif_path(img_metadata_ms[img], metadata_path, 'ms')
    return img_metadata_pan, img_metadata_ms

def add_names_to_metadata_dict(metadata_dictionary, list_of_area_names):
    """Adds area names to metadata dictionary"""
    for image_name in metadata_dictionary.keys():
        for area_name in list_of_area_names:
            if area_name in image_name:
                metadata_dictionary[image_name]['area_name'] = area_name
    return metadata_dictionary

def dict_to_df(img_metadata_dict):
    """Converts dictionary to Pandas DataFrame with correct data types."""
    img_metadata_df = pd.DataFrame(img_metadata_dict).transpose()
    img_metadata_df = img_metadata_df.astype(
        {'bitsPerPixel': 'category', 
         'cloudCover': 'float', 
         'datum': 'category', 
         'imageFileFormat': 'category',
         'imageTypeSize': 'category',
         'imagingTilingType': 'category',
         'isDynamicRangeAdjusted': 'bool', 
         'isMosaic': 'bool', 
         'mapProjection': 'category',
         'mapProjectionUnit': 'category',
         'mapProjectionZone': 'category', 
         'mergingAlgorithm': 'category',
         'mergedBand': 'category',
         'offNadirAngle': 'float',
         'pixelHeight': 'float',
         'pixelWidth': 'float',
         'processingLevel': 'category',
         'resamplingKernel': 'category',
         'sensorVehicle': 'category', 
         'sunAzimuth': 'float',
         'sunElevation': 'float', 
         'n_bands': 'int8', 
         'n_products': 'int8',
         'n_strips': 'int8'
    })
    
    # Converting the datatypes of the bands to category
    for i in range(20): #20 is just a number much higher than number of bands to be safe
        if str('band' + str(i)) in img_metadata_df.columns:
            img_metadata_df = img_metadata_df.astype({str('band'+str(i)): 'category'})
        
    img_metadata_df['earliestAcquisitionTime'] = pd.to_datetime(img_metadata_df['earliestAcquisitionTime'])
    img_metadata_df['latestAcquisitionTime'] = pd.to_datetime(img_metadata_df['latestAcquisitionTime'])
    img_metadata_df['productionDate'] = pd.to_datetime(img_metadata_df['productionDate'])
    img_metadata_df['updateDate'] = pd.to_datetime(img_metadata_df['updateDate'])

    # Adding unique IDs of dtype int (useful in TensorFlow)
    img_metadata_df['int_uid'] = list(range(len(img_metadata_df.index)))
    return img_metadata_df