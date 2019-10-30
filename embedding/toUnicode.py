import  codecs
def toUnicode(file1,file2):
    with open(file1, "r", encoding="gbk") as f:
        results = f.readlines()
    # 开始遍历读取结果，并写到新的文件中
    with codecs.open(file2, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result)
    print("转码成功！转码后文件为:", file2)
toUnicode("glove.840B.300d.txt","glove.840B.300.txt")
