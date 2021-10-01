import os
from tqdm import tqdm
from pprint import pprint
from PIL.ExifTags import TAGS
from PIL import Image
import numpy as np
from functools import reduce
from collections import OrderedDict
from extract_exif import get_exif, get_files_path
import cv2
from tqdm import tqdm
from functools import reduce
from pprint import pprint
import traceback
import sys

sys.path.append("openpyxl")
sys.path.append("et_xmlfile")

def database_investigate(db_dir_path, investigate_dict) :  
    instance_name_list = os.listdir(db_dir_path) #[instance_name for instance_name in os.listdir(db_dir_path)  if instance_name  not in ["Galaxy 10+", "iPhone 8", "iPhone 6", "Galaxy 8+", "Galaxy A30", "Galaxy A7(2018)", "Galaxy Note9", "Galaxy Note9 SM-N960N"]]  TODO : remove 제거 
    instance_path_list = [os.path.join(db_dir_path, instance_path) for instance_path in  instance_name_list] # 모든 인스턴스 디렉토리의 PATH 
    total_dict = investigate_dict["total"]

    total_dict["total_image"] = len(get_files_path(db_dir_path)) # 총 이미지 수

    for instance_path in tqdm(instance_path_list, desc = "instance") : 
        image_file_path_list = get_files_path(instance_path) # 해당 인스턴스의 모든 이미지의 path

        instance_dir_name = os.path.basename(instance_path)

        dark_image_dir_path = os.path.join(instance_path, "dark") # dark 디렉토리 path 
        flatfield_image_dir_path = os.path.join(instance_path, "flatfield") # flatfield 디렉토리 path
        natural_image_dir_path = os.path.join(instance_path, "natural") # natural 디렉토리 path
        complex_image_dir_path = os.path.join(natural_image_dir_path, "complex")
        simple_image_dir_path = os.path.join(natural_image_dir_path, "simple")
        indoor_image_dir_path_list = [os.path.join(image_dir_path, "실내") for image_dir_path in [complex_image_dir_path, simple_image_dir_path]]
        outdoor_day_image_dir_path_list = [os.path.join(image_dir_path, "실외주간") for image_dir_path in [complex_image_dir_path, simple_image_dir_path]]
        outdoor_night_image_dir_path_list = [os.path.join(image_dir_path, "실외야간") for image_dir_path in [complex_image_dir_path, simple_image_dir_path]]

        instance_dict = investigate_dict[instance_dir_name]

        instance_dict["n_darks"] = len(get_files_path(dark_image_dir_path)) # 인스턴스의 dark image 수
        instance_dict["n_flatfields"] = len(get_files_path(flatfield_image_dir_path)) # 인스턴스의 flatfield image 수
        instance_dict["n_naturals"] = len(get_files_path(natural_image_dir_path))

        instance_dict["n_complexs"] = len(get_files_path(complex_image_dir_path)) # 인스턴스의 complex image 수
        instance_dict["n_simples"] = len(get_files_path(simple_image_dir_path)) # 인스턴스의 simple image 수

        instance_dict["n_indoors"] = reduce(lambda x, y : x  + y, map(lambda image_dir_path : len(get_files_path(image_dir_path)), indoor_image_dir_path_list)) # 인스턴스의 실내 image 수
        instance_dict["n_outdoor_days"] = reduce(lambda x, y : x  + y, map(lambda image_dir_path : len(get_files_path(image_dir_path)), outdoor_day_image_dir_path_list)) # 인스턴스의 실내주간 image 수
        instance_dict["n_outdoor_nights"] = reduce(lambda x, y : x  + y, map(lambda image_dir_path : len(get_files_path(image_dir_path)), outdoor_night_image_dir_path_list)) # 인스턴스의 실내야간 image 수

        instance_dict["n_images"] = len(image_file_path_list) # 인스턴스의 총 image 수
        instance_dict["n_etcs"] = instance_dict["n_images"] - (instance_dict["n_darks"] + instance_dict["n_flatfields"] + instance_dict["n_complexs"] + instance_dict["n_simples"])

        total_dict["n_dark"] += instance_dict["n_darks"]
        total_dict["n_flatfield"] += instance_dict["n_flatfields"]
        total_dict["n_natural"] += instance_dict["n_naturals"]

        total_dict["n_complex"] += instance_dict["n_complexs"]
        total_dict["n_simple"] += instance_dict["n_simples"]

        total_dict["n_indoor"] += instance_dict["n_indoors"]
        total_dict["n_outdoor_day"] += instance_dict["n_outdoor_days"]
        total_dict["n_outdoor_night"] += instance_dict["n_outdoor_nights"]
        total_dict["n_etc"] += instance_dict["n_etcs"]
        
        instance_dict["unnecessary"] = [];


        for image_file_path in image_file_path_list : # 해당 인스턴스의 각 이미지
            image_extension = os.path.splitext(image_file_path)[1]
            basename = os.path.basename(os.path.splitext(image_file_path)[0])


            try : 
                assert "HDR" not in basename, "Abnormal file ({})".format(image_file_path) # HDR이라는 글자가 포함된 경우
                assert image_extension == ".JPG" or image_extension == ".jpg", "extention is not jpg format({})".format(image_file_path) # 이미지 파일의 확장자가 jpg 또는 JPG인지 확인
            except : 
                instance_dict["unnecessary"].append(image_file_path)
                continue;

            try : 
                image_exif = get_exif(image_file_path) # 해당 이미지의 exif
            except Exception as err: 
                traceback.print_exc(); print(image_file_path)
            
            try : 
                model = image_exif["Model"].strip().replace("\0", "")
            except Exception as err: 
                traceback.print_exc(); print(image_file_path, image_exif)


            assert instance_dict["model"] == model, "Model does not match.({}, {}, {})".format(instance_dict["model"], model, image_file_path) # 인스턴스의 모델 확인

            """ TODO : 주석 제거
            try : # ExifImageWidth, ExifImageHeight tag가 없거나 있어도 제대로 된 이미지 크기 정보를 가지지 않은 경우가 있음.
                width = image_exif["ExifImageWidth"]
                height = image_exif["ExifImageHeight"]

                assert instance_dict["width"] == width, "Width does not match.({}, {}, {})".format(instance_dict["width"], width, image_file_path) # 인스턴스의 가로 크기 확인
                assert instance_dict["height"] == height, "Height does not match.({}, {}, {})".format(instance_dict["height"], height, image_file_path) # 인스턴스의 세로 크기 확인
            except : 
                img_array = np.fromfile(image_file_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                try : 
                    cv2_height = img.shape[0]
                    cv2_width = img.shape[1]
                except Exception as err: 
                    cv2_height = img.shape[0]
                    cv2_width = img.shape[1]

                assert instance_dict["width"] == cv2_width, "Width does not match.({}, {}, {})".format(instance_dict["width"], cv2_width, image_file_path) # 인스턴스의 가로 크기 확인
                assert instance_dict["height"] == cv2_height, "Height does not match.({}, {}, {})".format(instance_dict["height"], cv2_height, image_file_path) # 인스턴스의 세로 크기 확인
            """
    return investigate_dict

def make_excel(excel_dict, excel_path) : 
        wb = Workbook()
        ws = wb.active
        print(excel_dict.keys())
        ws.append([" "] + list(excel_dict[list(excel_dict.keys())[0]].keys()))

        for key, value in excel_dict.items() : 
            cols = [key]
            
            value.pop('unnecessary', None)
            for key, value in value.items() :     
                cols.append(value)
            
            ws.append(cols)
        wb.save(excel_path)

if __name__ == "__main__" :
    SMDB_PATH = "C:/Users/rlfalsgh95/source/repos/Source_Identification_Datasets/SMDB"
    DIGITAL_CAMERA_IMAGE_DIR_PATH = os.path.join(SMDB_PATH, "digital_camera_image")
    SMARTPHONE_IMAGE_DIR_PATH = os.path.join(SMDB_PATH, "smartphone_image")

    digital_dict = OrderedDict({"Canon_EOS-6D_0" : {"manufacturer" : "Canon", "model" : "Canon EOS 6D", "width" : 5472, "height" : 3648},
                    "Canon_EOS-6D_1" : {"manufacturer" : "Canon", "model" : "Canon EOS 6D", "width" : 5472, "height" : 3648},
                    "Canon_EOS-80D_0" : {"manufacturer" : "Canon", "model" : "Canon EOS 80D", "width" : 6000, "height" : 4000},
                    "Canon_EOS-500D_0" : {"manufacturer" : "Canon", "model" : "Canon EOS 500D", "width" : 4752, "height" : 3168},
                    "Canon_EOS-500D_1" : {"manufacturer" : "Canon", "model" : "Canon EOS 500D", "width" : 4752, "height" : 3168},
                    "Canon_EOS-550D_0" : {"manufacturer" : "Canon", "model" : "Canon EOS 550D", "width" : 3456, "height" : 2304},
                    "Canon_EOS-750D_0" : {"manufacturer" : "Canon", "model" : "Canon EOS 750D", "width" : 6000, "height" : 4000},
                    "Canon_EOS-750D_1" : {"manufacturer" : "Canon", "model" : "Canon EOS 750D", "width" : 2976, "height" : 1984},
                    "Nikon_D7000_0" : {"manufacturer" : "NIKON CORPORATION", "model" : "NIKON D7000", "width" : 4928, "height" : 3264},
                    "Canon_PowerShot-SX210-IS_0" : {"manufacturer" : "Canon", "model" : "Canon PowerShot SX210 IS", "width" : 4320, "height" : 2432},
                    "Olympus_E-M10MarkII_0" : {"manufacturer" : "OLYMPUS CORPORATION", "model" : "E-M10MarkII", "width" : 4608, "height" : 3456},
                    "Sigma_DP1-Merrill_0" : {"manufacturer" : "SIGMA", "model" : "SIGMA DP1 Merrill", "width" : 4000, "height" : 2667},
                    "total" : {"n_dark" : 0, "n_flatfield" : 0, "n_complex" : 0, "n_simple" : 0, "n_indoor" : 0, "n_outdoor_day" : 0, "n_outdoor_night" : 0, "n_etc" : 0, "n_natural" : 0}
                    })

    smartphone_dict = OrderedDict({"Galaxy 8+" : {"manufacturer" : "samsung", "model" : "SM-G955N", "width" : 4032, "height" : 3024},
                       "Galaxy 10+" : {"manufacturer" : "samsung", "model" : "SM-G975N", "width" : 4032, "height" : 1908},
                       "Galaxy A5" : {"manufacturer" : "samsung", "model" : "SM-A500S", "width" : 3264, "height" : 2448},
                       "Galaxy A7(2018)" : {"manufacturer" : "samsung", "model" : "SM-A750N", "width" : 5664, "height" : 4248},
                       "Galaxy A8" : {"manufacturer" : "samsung", "model" : "SM-A530N", "width" : 2304, "height" : 1728},
                       "Galaxy A8(2018)" : {"manufacturer" : "samsung", "model" : "SM-A530N", "width" : 4608, "height" : 3456},
                       "Galaxy A8 Star" : {"manufacturer" : "samsung", "model" : "SM-G885S", "width" : 4608, "height" : 3456},
                       "Galaxy A30" : {"manufacturer" : "samsung", "model" : "SM-A305N", "width" : 4608, "height" : 3456},
                       "Galaxy J5" : {"manufacturer" : "samsung", "model" : "SM-J500N0", "width" : 4128, "height" : 2322},
                       "Galaxy J7" : {"manufacturer" : "samsung", "model" : "SM-J710K", "width" : 4128, "height" : 3096},
                       "Galaxy Note1" : {"manufacturer" : "samsung", "model" : "SHV-E160L", "width" : 3264, "height" : 2448},
                       "Galaxy Note5" : {"manufacturer" : "samsung", "model" : "SM-N920S", "width" : 5312, "height" : 2988},
                       "Galaxy Note9" : {"manufacturer" : "samsung", "model" : "SM-N960N", "width" : 4032, "height" : 1960},
                       "Galaxy Note9 SM-N960N" : {"manufacturer" : "samsung", "model" : "SM-N960N", "width" : 4032, "height" : 3024},
                       "Galaxy S4 LTEA" : {"manufacturer" : "samsung", "model" : "SHV-E330S", "width" : 4128, "height" : 2322},
                       "Galaxy S6" : {"manufacturer" : "samsung", "model" : "SM-G920K", "width" : 5312, "height" : 2988},
                       "Galaxy S7" : {"manufacturer" : "samsung", "model" : "SM-G930S", "width" : 4032, "height" : 3024},
                       "Galaxy S9" : {"manufacturer" : "samsung", "model" : "SM-G960N", "width" : 4032, "height" : 3024},
                       "Galaxy S 10 5g" : {"manufacturer" : "samsung", "model" : "SM-G977N", "width" : 4032, "height" : 3024},
                       "Galaxy S 10E" : {"manufacturer" : "samsung", "model" : "SM-G970N", "width" : 4032, "height" : 3024},
                       "Galaxy S7 Edge" : {"manufacturer" : "samsung", "model" : "SM-G935K", "width" : 4032, "height" : 3024},

                       "iPhone 6" : {"manufacturer" : "Apple", "model" : "iPhone 6", "width" : 3264, "height" : 2448},
                       "iPhone 8" : {"manufacturer" : "Apple", "model" : "iPhone 8", "width" : 4032, "height" : 3024},

                       "LG G2" : {"manufacturer" : "LG Electronics", "model" : "LG-F320L", "width" : 4160, "height" : 2340},
                       "LG G3 cat6" : {"manufacturer" : "LG Electronics", "model" : "LG-F460L", "width" : 4160, "height" : 2340},
                       "LG G6" : {"manufacturer" : "LG Electronics", "model" : "LGM-G600L", "width" : 4160, "height" : 3120},
                       "LG Q9 5g" : {"manufacturer" : "LG Electronics", "model" : "LM-Q925S", "width" : 4640, "height" : 2210},
                       "LG Xscreen" : {"manufacturer" : "LG Electronics", "model" : "LG-F650L", "width" : 4160, "height" : 2340},

                       "Hongmi note 8 pro" : {"manufacturer" : "Xiaomi", "model" : "Redmi Note 8 Pro", "width" : 4624, "height" : 3472},
                       "total" : {"n_dark" : 0, "n_flatfield" : 0, "n_complex" : 0, "n_simple" : 0, "n_indoor" : 0, "n_outdoor_day" : 0, "n_outdoor_night" : 0, "n_etc" : 0, "n_natural" : 0}
                })

    # digital_investigate_dict = database_investigate(DIGITAL_CAMERA_IMAGE_DIR_PATH, digital_dict)
    # np.save("digital_investigate.npy", digital_investigate_dict); pprint(digital_investigate_dict)

    