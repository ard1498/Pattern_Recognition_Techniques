# import matplotlib.pyplot as plt
import cv2

def main():
    # X = [-1,-1,1,1,-1]
    # Y = [-1,1,1,-1,-1]
    # plt.plot(X, Y, color='blue')
    # plt.show()

    image = cv2.imread('C:\\Users\\ANIRUDH\\Desktop\\SEM6\\ImageProcessingLab\\blastoise.png')
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(image)

if __name__ == '__main__':
    main()
