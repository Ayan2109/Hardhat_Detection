from zipfile import ZipFile
file_name_data = "../../HardHat_Dataset.zip"
file_name_train = "../../HardHat_Test_Images.zip"


if __name__ == '__main__':
	with ZipFile(file_name_data, 'r') as zip:

		print("extracting training data files....")
		zip.extractall('../detections')
		print("Done!")


	with ZipFile(file_name_train, 'r') as zip:

		print("extracting testing data files..")
		zip.extractall('../test_data')
		print("Done!")



