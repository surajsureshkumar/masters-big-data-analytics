import pandas as pd

def decision_tree(filename):
	data=pd.read_csv(filename)
	data['Shagginess'] = data['HairLn'] - data['BangLn']
	data['ApeFactor'] = data['Reach'] - data['Ht']
	output = []
	for index, row in data.iterrows():
		if row['BangLn'] <= 5.5:
			if row['HairLn'] <= 12.0:
				if row['ApeFactor'] <= 6.0:
					if row['TailLn'] <= 2.0:
						print(1)
						output.append(1)
					else:
						if row['Age'] <= 15.0:
							print(1)
							output.append(1)
						else:
							print(-1)
							output.append(-1)
				else:
					print(1)
					output.append(1)
			else:
				print(1)
				output.append(1)
		else:
			if row['ApeFactor'] <= 4.0:
				print(-1)
				output.append(-1)
			else:
				if row['Age'] <= 64.0:
					if row['Ht'] <= 181.0:
						print(1)
						output.append(1)
					else:
						print(-1)
						output.append(-1)
				else:
					print(-1)
					output.append(-1)

	data['Predicted_ClassID'] = output
	return data


if __name__ == "__main__":
	data = decision_tree("Abominable_VALIDATION_Data_FOR_STUDENTS_v772_2231.csv")
	data.to_csv("HW_06_Kothari_Sureshkumar_MyClassifications.csv")