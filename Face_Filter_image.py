from PIL import Image, ImageDraw
import face_recognition, math, numpy, cv2

def Bat_Filter(url):
    '''
    url: local path of the image
    '''
    image = face_recognition.load_image_file(url)
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_loco = face_recognition.face_locations(image)
    bat = Image.open('./Filter_image/bat.png')
    angle = (face_landmarks_list[0]['right_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1]) / (face_landmarks_list[0]['right_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])
    bat_angle = -math.atan(angle) * 180 / math.pi
    bat_angle_rad = math.radians(bat_angle)
    Angle_matrix = numpy.array([[math.cos(bat_angle_rad), -math.sin(bat_angle_rad)], [math.sin(bat_angle_rad), math.cos(bat_angle_rad)]])
    Vector1 = numpy.array([41 - bat.size[0]//2, bat.size[1]//2 - 307])
    Vector2 = numpy.array([178 - bat.size[0]//2, bat.size[1]//2 -307])
    res1 = numpy.dot(Angle_matrix, Vector1)
    res2 = numpy.dot(Angle_matrix, Vector2)

    pre_bat_size = bat.size
    bat = bat.rotate(bat_angle, expand=True)
    aft_bat_size = bat.size
    bat_delta = (face_landmarks_list[0]['right_eye'][0][0] - face_landmarks_list[0]['left_eye'][0][0])/(res2[0] - res1[0])
    print('angle: ', bat_angle)
    bat = bat.resize((int(bat.size[0] * bat_delta), int(bat.size[1] * bat_delta)))
    center_eye = [face_landmarks_list[0]['left_eye'][0][0] + (face_landmarks_list[0]['left_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])/2, face_landmarks_list[0]['left_eye'][0][1] + (face_landmarks_list[0]['left_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1])]


    pil_image = Image.fromarray(image)
    Vector3 = numpy.array([75* bat_delta  - bat.size[0]//2 / aft_bat_size[0]*pre_bat_size[0] ,  bat.size[1]//2/ aft_bat_size[1]*pre_bat_size[1] - 320 * bat_delta])
    res3 = numpy.dot(Angle_matrix, Vector3)
    test = [bat.size[0]//2 + res3[0], bat.size[1]//2 - res3[1]]
    LocoBat = [int(center_eye[0] - (bat.size[0]//2 + res3[0])) , int(center_eye[1] - (bat.size[1]//2 - res3[1]))]

    # DEBUGGING
    # print(Vector3)
    # print(res3)
    # print(test)
    # print(bat.size[0]//2, bat.size[1]//2)
    # d = ImageDraw.Draw(bat, 'RGBA')
    # d.point(center_eye, fill=(150, 0, 0))
    # d.ellipse((test[0] - 5, test[1] - 5, test[0] + 5,test[1] + 5), fill='#f00')
    pil_image.paste(bat, LocoBat, bat)
    return pil_image

def Cry_filter(url):
    '''
    url: local path of the image
    '''
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(url)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_loco = face_recognition.face_locations(image)


    bat = Image.open('./Filter_image/cry.png')
    angle = (face_landmarks_list[0]['right_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1]) / (face_landmarks_list[0]['right_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])
    bat_angle = -math.atan(angle) * 180 / math.pi
    bat_angle_rad = math.radians(bat_angle)
    Angle_matrix = numpy.array([[math.cos(bat_angle_rad), -math.sin(bat_angle_rad)], [math.sin(bat_angle_rad), math.cos(bat_angle_rad)]])
    Vector1 = numpy.array([31 - bat.size[0]//2, bat.size[1]//2 - 151])
    Vector2 = numpy.array([706 - bat.size[0]//2, bat.size[1]//2 -151])
    res1 = numpy.dot(Angle_matrix, Vector1)
    res2 = numpy.dot(Angle_matrix, Vector2)

    pre_bat_size = bat.size
    bat = bat.rotate(bat_angle, expand=True)
    aft_bat_size = bat.size
    bat_delta = (face_landmarks_list[0]['right_eye'][0][0] - face_landmarks_list[0]['left_eye'][0][0])/(res2[0] - res1[0])
    print('angle: ', bat_angle)
    bat = bat.resize((int(bat.size[0] * bat_delta), int(bat.size[1] * bat_delta)))
    center_eye = [face_landmarks_list[0]['left_eye'][0][0] + (face_landmarks_list[0]['left_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])/2, face_landmarks_list[0]['left_eye'][0][1] + (face_landmarks_list[0]['left_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1])]


    pil_image = Image.fromarray(image)

    Vector3 = numpy.array([149* bat_delta  - bat.size[0]//2 / aft_bat_size[0]*pre_bat_size[0] ,  bat.size[1]//2/ aft_bat_size[1]*pre_bat_size[1] - 141 * bat_delta])
    res3 = numpy.dot(Angle_matrix, Vector3)

    test = [bat.size[0]//2 + res3[0], bat.size[1]//2 - res3[1]]
    LocoBat = [int(center_eye[0] - (bat.size[0]//2 + res3[0])) , int(center_eye[1] - (bat.size[1]//2 - res3[1]))]

    # DEBUGGING
    # print(Vector3)
    # print(res3)
    # print(test)
    # print(bat.size[0]//2, bat.size[1]//2)
    # d = ImageDraw.Draw(bat, 'RGBA')
    # d.point(center_eye, fill=(150, 0, 0))
    # d.ellipse((test[0] - 5, test[1] - 5, test[0] + 5,test[1] + 5), fill='#f00')
    pil_image.paste(bat, LocoBat, bat)
    return pil_image

def Tear_filter(url):
    '''
    url: local path of the image
    '''
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(url)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_loco = face_recognition.face_locations(image)


    bat = Image.open('./Filter_image/cry2.png')
    angle = (face_landmarks_list[0]['right_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1]) / (face_landmarks_list[0]['right_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])
    bat_angle = -math.atan(angle) * 180 / math.pi
    bat_angle_rad = math.radians(bat_angle)
    Angle_matrix = numpy.array([[math.cos(bat_angle_rad), -math.sin(bat_angle_rad)], [math.sin(bat_angle_rad), math.cos(bat_angle_rad)]])
    Vector1 = numpy.array([190 - bat.size[0]//2, bat.size[1]//2 - 212])
    Vector2 = numpy.array([722 - bat.size[0]//2, bat.size[1]//2 -212])
    res1 = numpy.dot(Angle_matrix, Vector1)
    res2 = numpy.dot(Angle_matrix, Vector2)

    pre_bat_size = bat.size
    bat = bat.rotate(bat_angle, expand=True)
    aft_bat_size = bat.size
    bat_delta = (face_landmarks_list[0]['right_eye'][0][0] - face_landmarks_list[0]['left_eye'][0][0])/(res2[0] - res1[0])
    print('angle: ', bat_angle)
    bat = bat.resize((int(bat.size[0] * bat_delta), int(bat.size[1] * bat_delta)))
    center_eye = [face_landmarks_list[0]['left_eye'][0][0] + (face_landmarks_list[0]['left_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])/2, face_landmarks_list[0]['left_eye'][0][1] + (face_landmarks_list[0]['left_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1])]


    pil_image = Image.fromarray(image)

    Vector3 = numpy.array([230* bat_delta  - bat.size[0]//2 / aft_bat_size[0]*pre_bat_size[0] ,  bat.size[1]//2/ aft_bat_size[1]*pre_bat_size[1] - 282 * bat_delta])
    res3 = numpy.dot(Angle_matrix, Vector3)

    test = [bat.size[0]//2 + res3[0], bat.size[1]//2 - res3[1]]
    LocoBat = [int(center_eye[0] - (bat.size[0]//2 + res3[0])) , int(center_eye[1] - (bat.size[1]//2 - res3[1]))]

    # DEBUGGING
    # print(Vector3)
    # print(res3)
    # print(test)
    # print(bat.size[0]//2, bat.size[1]//2)
    # d = ImageDraw.Draw(bat, 'RGBA')
    # d.point(center_eye, fill=(150, 0, 0))
    # d.ellipse((test[0] - 5, test[1] - 5, test[0] + 5,test[1] + 5), fill='#f00')
    pil_image.paste(bat, LocoBat, bat)
    return pil_image

def SunGlass_filter(url):
    '''
    url: local path of the image
    '''
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(url)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_loco = face_recognition.face_locations(image)


    bat = Image.open('./Filter_image/glasses.png')
    angle = (face_landmarks_list[0]['right_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1]) / (face_landmarks_list[0]['right_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])
    bat_angle = -math.atan(angle) * 180 / math.pi
    bat_angle_rad = math.radians(bat_angle)
    Angle_matrix = numpy.array([[math.cos(bat_angle_rad), -math.sin(bat_angle_rad)], [math.sin(bat_angle_rad), math.cos(bat_angle_rad)]])
    Vector1 = numpy.array([21 - bat.size[0]//2, bat.size[1]//2 - 88])
    Vector2 = numpy.array([292 - bat.size[0]//2, bat.size[1]//2 -88])
    res1 = numpy.dot(Angle_matrix, Vector1)
    res2 = numpy.dot(Angle_matrix, Vector2)

    pre_bat_size = bat.size
    bat = bat.rotate(bat_angle, expand=True)
    aft_bat_size = bat.size
    bat_delta = (face_landmarks_list[0]['right_eye'][0][0] - face_landmarks_list[0]['left_eye'][0][0])/(res2[0] - res1[0])
    print('angle: ', bat_angle)
    bat = bat.resize((int(bat.size[0] * bat_delta), int(bat.size[1] * bat_delta)))
    center_eye = [face_landmarks_list[0]['left_eye'][0][0] + (face_landmarks_list[0]['left_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])/2, face_landmarks_list[0]['left_eye'][0][1] + (face_landmarks_list[0]['left_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1])]


    pil_image = Image.fromarray(image)

    Vector3 = numpy.array([121* bat_delta  - bat.size[0]//2 / aft_bat_size[0]*pre_bat_size[0] ,  bat.size[1]//2/ aft_bat_size[1]*pre_bat_size[1] - 106 * bat_delta])
    res3 = numpy.dot(Angle_matrix, Vector3)

    test = [bat.size[0]//2 + res3[0], bat.size[1]//2 - res3[1]]
    LocoBat = [int(center_eye[0] - (bat.size[0]//2 + res3[0])) , int(center_eye[1] - (bat.size[1]//2 - res3[1]))]

    # DEBUGGING
    # print(Vector3)
    # print(res3)
    # print(test)
    # print(bat.size[0]//2, bat.size[1]//2)
    # d = ImageDraw.Draw(bat, 'RGBA')
    # d.point(center_eye, fill=(150, 0, 0))
    # d.ellipse((test[0] - 5, test[1] - 5, test[0] + 5,test[1] + 5), fill='#f00')
    pil_image.paste(bat, LocoBat, bat)
    return pil_image

def Sharingan_filter(url):
    '''
    url: local path of the image
    '''
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(url)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_loco = face_recognition.face_locations(image)


    bat = Image.open('./Filter_image/sharingan.png')
    angle = (face_landmarks_list[0]['right_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1]) / (face_landmarks_list[0]['right_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])
    bat_angle = -math.atan(angle) * 180 / math.pi
    bat_angle_rad = math.radians(bat_angle)
    Angle_matrix = numpy.array([[math.cos(bat_angle_rad), -math.sin(bat_angle_rad)], [math.sin(bat_angle_rad), math.cos(bat_angle_rad)]])
    Vector1 = numpy.array([86 - bat.size[0]//2, bat.size[1]//2 - 70])
    Vector2 = numpy.array([360 - bat.size[0]//2, bat.size[1]//2 -70])
    res1 = numpy.dot(Angle_matrix, Vector1)
    res2 = numpy.dot(Angle_matrix, Vector2)

    pre_bat_size = bat.size
    bat = bat.rotate(bat_angle, expand=True)
    aft_bat_size = bat.size
    bat_delta = (face_landmarks_list[0]['right_eye'][0][0] - face_landmarks_list[0]['left_eye'][0][0])/(res2[0] - res1[0])
    print('angle: ', bat_angle)
    bat = bat.resize((int(bat.size[0] * bat_delta), int(bat.size[1] * bat_delta)))
    center_eye = [face_landmarks_list[0]['left_eye'][0][0] + (face_landmarks_list[0]['left_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])/2, face_landmarks_list[0]['left_eye'][0][1] + (face_landmarks_list[0]['left_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1])]


    pil_image = Image.fromarray(image)

    Vector3 = numpy.array([135* bat_delta  - bat.size[0]//2 / aft_bat_size[0]*pre_bat_size[0] ,  bat.size[1]//2/ aft_bat_size[1]*pre_bat_size[1] - 72 * bat_delta])
    res3 = numpy.dot(Angle_matrix, Vector3)

    test = [bat.size[0]//2 + res3[0], bat.size[1]//2 - res3[1]]
    LocoBat = [int(center_eye[0] - (bat.size[0]//2 + res3[0])) , int(center_eye[1] - (bat.size[1]//2 - res3[1]))]

    # DEBUGGING
    # print(Vector3)
    # print(res3)
    # print(test)
    # print(bat.size[0]//2, bat.size[1]//2)
    # d = ImageDraw.Draw(bat, 'RGBA')
    # d.point(center_eye, fill=(150, 0, 0))
    # d.ellipse((test[0] - 5, test[1] - 5, test[0] + 5,test[1] + 5), fill='#f00')
    pil_image.paste(bat, LocoBat, bat)
    return pil_image

def SkiMask_filter(url):
    '''
    url: local path of the image
    '''
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(url)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_loco = face_recognition.face_locations(image)


    bat = Image.open('./Filter_image/skimask2.png')
    angle = (face_landmarks_list[0]['right_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1]) / (face_landmarks_list[0]['right_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])
    bat_angle = -math.atan(angle) * 180 / math.pi
    bat_angle_rad = math.radians(bat_angle)
    Angle_matrix = numpy.array([[math.cos(bat_angle_rad), -math.sin(bat_angle_rad)], [math.sin(bat_angle_rad), math.cos(bat_angle_rad)]])
    Vector1 = numpy.array([40 - bat.size[0]//2, bat.size[1]//2 - 190])
    Vector2 = numpy.array([161 - bat.size[0]//2, bat.size[1]//2 -190])
    res1 = numpy.dot(Angle_matrix, Vector1)
    res2 = numpy.dot(Angle_matrix, Vector2)

    pre_bat_size = bat.size
    bat = bat.rotate(bat_angle, expand=True)
    aft_bat_size = bat.size
    bat_delta = (face_landmarks_list[0]['right_eye'][0][0] - face_landmarks_list[0]['left_eye'][0][0])/(res2[0] - res1[0])
    print('angle: ', bat_angle)
    bat = bat.resize((int(bat.size[0] * bat_delta), int(bat.size[1] * bat_delta)))
    center_eye = [face_landmarks_list[0]['left_eye'][0][0] + (face_landmarks_list[0]['left_eye'][3][0] - face_landmarks_list[0]['left_eye'][0][0])/2, face_landmarks_list[0]['left_eye'][0][1] + (face_landmarks_list[0]['left_eye'][3][1] - face_landmarks_list[0]['left_eye'][0][1])]


    pil_image = Image.fromarray(image)

    Vector3 = numpy.array([74* bat_delta  - bat.size[0]//2 / aft_bat_size[0]*pre_bat_size[0] ,  bat.size[1]//2/ aft_bat_size[1]*pre_bat_size[1] - 201 * bat_delta])
    res3 = numpy.dot(Angle_matrix, Vector3)

    test = [bat.size[0]//2 + res3[0], bat.size[1]//2 - res3[1]]
    LocoBat = [int(center_eye[0] - (bat.size[0]//2 + res3[0])) , int(center_eye[1] - (bat.size[1]//2 - res3[1]))]

    # DEBUGGING
    # print(Vector3)
    # print(res3)
    # print(test)
    # print(bat.size[0]//2, bat.size[1]//2)
    # d = ImageDraw.Draw(bat, 'RGBA')
    # d.point(center_eye, fill=(150, 0, 0))
    # d.ellipse((test[0] - 5, test[1] - 5, test[0] + 5,test[1] + 5), fill='#f00')
    pil_image.paste(bat, LocoBat, bat)
    return pil_image