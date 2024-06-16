import os
import pickle
import cv2
import numpy as np
import mediapipe as mp

# Define variables
width, height = 1280, 720
folderPath = "Presentation"
pathImages = sorted(os.listdir(folderPath), key=len)
imgNumber = 0
heightSmall, widthSmall = int(120 * 2), int(213 * 2)
threshold = 500
buttonPressed = False
buttonCounter = 0
buttonDelay = 11
line = [[]]
lineNumber = 0
lineStart = False

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'Left', 1: 'Right', 2: 'Pointer', 3: 'Draw', 4: 'Erase', 5: 'Do Nothing'}

# Create camera object setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)


# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    # Import presentation images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    hands_result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if hands_result.multi_hand_landmarks:
        hand_landmarks = hands_result.multi_hand_landmarks[0]  # We only consider the first detected hand
        mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        lmList = [[int(landmark.x * width), int(landmark.y * height)] for landmark in hand_landmarks.landmark]
        x_ = []
        y_ = []
        data_aux = []

        # Collect coordinates
        for x, y in lmList:
            x_.append(x / width)  # Normalize by width
            y_.append(y / height)  # Normalize by height

        for x, y in lmList:
            data_aux.append(x / width - min(x_))
            data_aux.append(y / height - min(y_))

        # Ensure data_aux has the correct length (42 features)
        if len(data_aux) != 42:
            data_aux.extend([0] * (42 - len(data_aux)))  # Padding if less than 42 features
        elif len(data_aux) > 42:
            data_aux = data_aux[:42]  # Trimming if more than 42 features

        # Duplicate data_aux to match the model's expected input of 84 features
        data_aux = np.hstack((data_aux, data_aux))

        # Make a prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        centerX, centerY = int(np.mean([x for x, y in lmList])), int(np.mean([y for x, y in lmList]))

        # Display the predicted character on the image
        cv2.putText(img, predicted_character, (centerX - 80, centerY - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Constrain for drawing
        # xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, 1980]))
        xVal = int(np.interp(lmList[8][0], [120, width // 2], [-100, 1980]))
        yVal = int(np.interp(lmList[8][1], [0, height - 300], [0, 1080]))
        indexFinger = (xVal, yVal)

        if centerY <= threshold:  # if hand Y axis is above the threshold line
            # Gesture left
            if predicted_character == 'Left':
                lineStart = False
                print("Left")
                if imgNumber > 0 and buttonPressed is False:
                    buttonPressed = True
                    line = [[]]
                    lineNumber = 0
                    imgNumber -= 1

            # Gesture right
            if predicted_character == 'Right':
                lineStart = False
                print("Right")
                if imgNumber < len(pathImages) - 1 and buttonPressed is False:
                    buttonPressed = True
                    line = [[]]
                    lineNumber = 0
                    imgNumber += 1

        # Gesture pointer
        if predicted_character == 'Pointer':
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            lineStart = False

        # Gesture draw
        if predicted_character == 'Draw':
            if lineStart is False:
                lineStart = True
                lineNumber += 1
                line.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            line[lineNumber].append(indexFinger)
        else:
            lineStart = False

        # Gesture erase
        if predicted_character == 'Erase':
            lineStart = False
            if line and buttonPressed is False and lineNumber >= 0:
                buttonPressed = True
                line.pop(-1)
                lineNumber -= 1

    else:
        lineStart = False

    # Handle the button delay
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range(len(line)):
        for j in range(len(line[i])):
            if j != 0:
                cv2.line(imgCurrent, line[i][j - 1], line[i][j], (0, 0, 255), 12)

    # Resizing camera on slide
    imgSmall = cv2.resize(img, (widthSmall, heightSmall))
    h, w, _ = imgCurrent.shape
    imgCurrent[h - heightSmall:h, w - widthSmall:w] = imgSmall

    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
