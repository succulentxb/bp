import img_recog
import sys

if __name__ == "__main__":
    # 1 hidden layer
    for neu_num in range(10, 150, 10):
        sizes = [784, neu_num, 14]
        for learning_rate in range(5, 50, 5):
            train_result = img_recog.network_train_quick(sizes, learning_rate / 100)
            file = open("./img_para_test_1", "a+")
            file.write("--------------------------------------------\n")
            file.write("sizes: " + str(sizes) + ", learning_rate: " + str(learning_rate / 100) + "\n")
            file.write("train error: " + str(train_result['error']) + "%\n")
            file.write("--------------------------------------------\n")
            file.close()