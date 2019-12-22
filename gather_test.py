import shutil

file_num = 0
for img_i in range(200, 255):
    for word_i in range(1, 15):
        scr_path = './TRAIN/' + str(word_i) + '/' + str(img_i) + '.bmp'
        dst_path = './TEST/' + str(file_num) + '.bmp'
        shutil.copy(scr_path, dst_path)
        file_num += 1
        if file_num > 762:
            print('moving done')
            exit()