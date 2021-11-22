import cv2
import numpy as np

w1 = np.load('weightbias/w1.npy')  # weight layer 1
b1 = np.load('weightbias/b1.npy')  # bias layer 1
w2 = np.load('weightbias/w2.npy')  # weight layer 2
b2 = np.load('weightbias/b2.npy')  # bias layer 2


def normalize(data, minimal, maximal):
    y = np.float32(data)
    for x in range(data.size):
        y[x] = 2*(data[x]-minimal)/(maximal-minimal) - 1
    return y


def denormalize(data, minimal, maximal):
    y = np.float32(data)
    for x in range(data.size):
        y[x] = 0.5*(data[x]+1)*(maximal-minimal) + minimal
    return y


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


def dsigmoid(x):
    y = (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))


def tansig(x):
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y


def dtansig(x):
    y = 1 - ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))**2
    return y


# Definisi fungsi simulasi real-time
def putTextInRectange(image, text, position):
    x, y = position[0], position[1]
    (txt_w, txt_h), bline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.8, 1)
    cv2.rectangle(image, (x-2, y-6), (x+200, y-35+bline),
                  (0, 255, 255), cv2.FILLED)
    cv2.putText(image, text, (x, y-10),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.7, (0, 0, 0), 2)


font = cv2.FONT_HERSHEY_SIMPLEX


def drawBoxAndWriteText(image, findfaces):
    for (x, y, w, h) in findfaces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 4)
        cv2.putText(image, 'face: '+str(w),
                    (10, 30), font, 0.6, (0, 0, 255), 2)

        # Neural net
        i = np.array([h])
        norm = normalize(i, 0, 480)
        h_layer = sigmoid(
            np.add(np.matmul(w1, norm[0].reshape(1, 1)), b1))  # F1(w1*x+b1)
        o_layer = tansig(np.add(np.matmul(w2, h_layer), b2))  # F2(w2*x+b2)
        denorm = denormalize(o_layer, 0, 480)

        out = np.round(denorm[0, 0], 2)
        putTextInRectange(image, 'Distance: '+str(out)+' cm ', (x, y))


def drawBox(image, FindEyes):
    for (ex, ey, ew, eh) in FindEyes:
        cv2.rectangle(image, (ex, ey), (ex+ew, ey+eh), (0, 255, 153), 2)
        cv2.putText(image, 'LE: '+str(ew),
                    (10, 50), font, 0.6, (0, 0, 255), 2)
        cv2.putText(image, 'RE: '+str(eh),
                    (10, 70), font, 0.6, (0, 0, 255), 2)


# Simulasi Real-time
cap = cv2.VideoCapture(1)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('distance_measurement.mp4', fourcc, 30.0, (640, 480))

width = cap.get(3)  # float `width`
height = cap.get(4)  # float `height
print(width, height)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
while(True):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        eyes = eye_cascade.detectMultiScale(gray, 1.2, 5)
        drawBoxAndWriteText(frame, faces)
        drawBox(frame, eyes)

        cv2.imshow('Distance', frame)
    else:
        print("Tangkapan gagal")
        break
    out.write(frame)
    keybrd = cv2.waitKey(1)
    if keybrd == ord('s'):
        cv2.imwrite('Jarak_wajah.jpg', frame)
    if keybrd == ord('q') or keybrd == 27:
        break
cap.release()
cv2.destroyAllWindows()
