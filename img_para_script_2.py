import img_recog
import sys

if __name__ == "__main__":    
    # 2 hidden layer
    for neu_num_1 in range(10, 110, 10):
        for neu_num_2 in range(10, 50, 10):
            sizes = [784, neu_num_1, neu_num_2, 14]
            for learning_rate in range(1, 20, 2):
                train_result = img_recog.network_train_quick(sizes, learning_rate / 100)
                file = open("./img_para_test_2", "a+")
                file.write("--------------------------------------------\n")
                file.write("sizes: " + str(sizes) + ", learning_rate: " + str(learning_rate / 100) + "\n")
                file.write("train error: " + str(train_result['error']) + "%\n")
                file.write("--------------------------------------------\n")
                file.close()