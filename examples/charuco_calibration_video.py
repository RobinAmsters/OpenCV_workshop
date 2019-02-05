import cv2.aruco
import pickle

def count_frames_manual(videoFileName):
	# initialize the total number of frames read
    total = 0
    video = cv2.VideoCapture(videoFileName)
    print('Counting frames')
 
	# loop over the frames of the video
    while True:
		# grab the current frame
        (grabbed, frame) = video.read()
	 
		# check to see if we have reached the end of the
		# video
        if not grabbed:
            break
 
		# increment the total number of frames read
        total += 1
 
	# return the total number of frames in the video file
    return total

def get_charuco_board(dictionary, save=False):
    board = cv2.aruco.CharucoBoard_create(3,3,.096,.048,dictionary)
    
    if save:
        #Dump the calibration board to a file
        img = board.draw((200*3,200*3))
        cv2.imwrite('charuco.png',img)
    
    return board

if __name__ == "__main__":

    # General parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = get_charuco_board(dictionary, False)
    videoFileName = 'calibration_data/calibration_big.mp4'
    
    #Start capturing images for calibration
    cap = cv2.VideoCapture(videoFileName)
    
    allCorners = []
    allIds = []
    decimator = 0
    
    n_frames = count_frames_manual(videoFileName)
    
    for i in range(n_frames):
    
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray,dictionary)
    
        if len(res[0])>0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%3==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])
    
            cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
    
        cv2.imshow('frame',gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        decimator+=1
        
        print('Calibrating: ' + str(i*100.0/n_frames) + "% of makers detected")
    
    imsize = gray.shape
    
    #Calibration fails for lots of reasons. Release the video if we do
    try:
        print("Calculating camera parameters, THIS MAY TAKE A VERY LONG TIME")
        cal = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
        pickle.dump(cal, open("camParams_USB.p", "wb" ))
        print('Calibration parameters saved')
    except:
        print('Calibration failed')
        cap.release()
    
    cap.release()
    cv2.destroyAllWindows()
