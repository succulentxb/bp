import sin_fitting as sf

if __name__ == "__main__":
    for i in range(1, 11):
        sizes = [1]
        for j in range(i):
            sizes.append(10)
        sizes.append(1)
        learning_rate = 0.05
        train_result = sf.network_train(sizes, learning_rate)
        file = open("./para_test_3", "a+")
        file.write("---------------------------\n")
        file.write("sizes: " + str(sizes) + ", learning rate: " + str(learning_rate) + "\n")
        file.write("train error: " + str(train_result['error']) + "\n")
        file.write("---------------------------\n")
        file.close()