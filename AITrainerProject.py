import cv2
import numpy as np
import time
import PoseModule as pm

# cap=cv2.VideoCapture("AITrainer/barbell biceps curl_1.mp4")
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera
  
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


detector=pm.poseDetector()
count=0
dir=0
pTime=0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    success, img=cap.read()
    img=cv2.resize(img,(1280,720))
    # img = cv2.imread("AITrainer/test.jpg")
    img=detector.findPose(img,False)
    lmList=detector.findPosition(img,False)
    # print(lmList)


    # Rep COunting bicep curl

    if len(lmList) != 0:
        #left arm
        angle_l=detector.findAngle(img,11,13,15)
        # right arm
        angle_r=detector.findAngle(img, 12, 14, 16)
        per_r=np.interp(angle_r,(46,175),(0,100))
        per_l = np.interp(angle_l, (20, 175), (0, 100))
        # print(angle_l,per_l)

        if((per_r==100) & (per_l==100)):
            if(dir==0):
                count+=0.5
                dir=1

        if((per_r==0) & (per_l==0)):
            if(dir==1):
                count+=0.5
                dir=0

        cv2.rectangle(img,(20,610),(300,690),(255,127,80),cv2.FILLED)
        cv2.putText(img,"Reps: "+str(int(count)),(45,670),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),4)

        cTime=time.time()
        fps= 1/(cTime - pTime)
        pTime=cTime
        cv2.putText(img, "FPS: "+str(int(fps)), (58, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


        #checking back posture
        back_angle1= detector.findAngle(img,12,24,26)
        back_angle2= detector.findAngle(img,11,23,25)

        # if((back_angle1!=0)&(back_angle2!=0)):
        #     if((back_angle1<155) | (back_angle2<155)| (back_angle1>185) | (back_angle2>185)):
        #         cv2.rectangle(img, (780, 1080), (530, 720), (255, 127, 80), cv2.FILLED)
        #         cv2.putText(img, "Straighten your back" , (45, 670), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)


    cv2.imshow('Image',img)

    # Display the frame in a window
    # cv2.imshow('Webcam', frame)
    cv2.waitKey(1)

    import cv2
    import numpy as np
    import PoseModule as pm
    from sklearn.svm import LinearSVC


    # Function to train the SVM model
    def train_svm(normalized_data, POSES):
        NUM_TEST = 10
        X_train = None
        y_train = None
        X_test = None
        y_test = None

        for pose, class_id in POSES.items():
            df = normalized_data[class_id]
            X_pose = df.to_numpy()
            y_pose = [class_id] * df.shape[0]

            X_pose_train = X_pose[:-NUM_TEST][:]
            y_pose_train = y_pose[:-NUM_TEST]
            X_pose_test = X_pose[-NUM_TEST:][:]
            y_pose_test = y_pose[-NUM_TEST:]

            if X_train is None:
                X_train = X_pose_train
            else:
                X_train = np.concatenate((X_train, X_pose_train), axis=0)

            if y_train is None:
                y_train = y_pose_train
            else:
                y_train = np.concatenate((y_train, y_pose_train))

            if X_test is None:
                X_test = X_pose_test
            else:
                X_test = np.concatenate((X_test, X_pose_test), axis=0)

            if y_test is None:
                y_test = y_pose_test
            else:
                y_test = np.concatenate((y_test, y_pose_test))

        clf = LinearSVC(C=1.0)
        clf.fit(X_train, y_train)

        # Return the trained model
        return clf


    # Train the SVM model
    # Replace 'normalized_data' and 'POSES' with your actual data
    clf = train_svm(normalized_data, POSES)

    # Initialize pose detector
    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0

    # Start capturing video
    cap = cv2.VideoCapture(0)  # Use the appropriate video source
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        # ... Your pose analysis code ...

        # Evaluate using the trained SVM model
        tests = list(zip(clf.predict(X_test), y_test))
        incorrect = [element for element in tests if element[0] != element[1]]
        accuracy = 1 - (len(incorrect) / len(tests))
        print('Ratio correct:', accuracy)

        # ... Display video frames and other real-time analysis ...

        cv2.imshow('Image', img)
        cv2.waitKey(1)

    # Release the camera and close windows when finished
    cap.release()
    cv2.destroyAllWindows()
