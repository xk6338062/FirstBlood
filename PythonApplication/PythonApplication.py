import requests
from bs4 import BeautifulSoup
import re
import os
import xlwt

class Lesson(object):	
	def __init__(self,_name,_price,_order):
		self.name = _name
		self.price = _price
		self.order = _order

def set_style(name,height,bold=False):
	style = xlwt.XFStyle()
	font = xlwt.Font()
	font.name = name
	font.bold = bold
	font.color_index = 4
	font.height = height
	style.font = font
	return style

def Excel(ls):
	f = xlwt.Workbook()
	sheet1 = f.add_sheet('数据',cell_overwrite_ok=True)
	row0 = ["序号","课程名称","售价","订阅数","名义收入","收入系数","实际收入","BOX归属"]
	style = set_style('Times New Roman',220,True)
	sheet1.write(0,0,row0[0])
	sheet1.write(0,1,row0[1])
	sheet1.write(0,2,row0[2])
	sheet1.write(0,3,row0[3])
	sheet1.write(0,4,row0[4])
	sheet1.write(0,5,row0[5])
	sheet1.write(0,6,row0[6])
	sheet1.write(0,7,row0[7])
	#写第一行
	for i in range(1,len(ls)):
		for j in range(0,len(row0)):
			if(j == 0):
				sheet1.write(i,j,i,style)
			elif(j == 1):
				sheet1.write(i,j,ls[i - 1].name,style)
			elif(j == 2):
				sheet1.write(i,j,ls[i - 1].price,style)
			elif(j == 3):
				sheet1.write(i,j,ls[i - 1].order,style)
			elif(j == 4):
				sheet1.write(i,j,ls[i - 1].order * ls[i - 1].price,style)
			elif(j == 5):
				sheet1.write(i,j,0.751664995,style)
			elif(j == 6):
				sheet1.write(i,j,ls[i - 1].order * ls[i - 1].price * 0.751664995,style)
			elif(j == 7):
				sheet1.write(i,j,ls[i - 1].order * ls[i - 1].price * 0.751664995 / 2,style)
	f.save('test.xls')

if __name__ == '__main__':
	url = "https://appbqa3jgpf2621.pc.xiaoe-tech.com/page/480624"
	response = requests.get(url)
	soup = BeautifulSoup(response.content, "html.parser")
	menu_tag = soup.find_all(class_="hot-item")
	texts = ""
	names = []
	lessons = []
	for menu in menu_tag:	
		words = menu.text.split('\n')
		if(len(words) > 6):			
			name = words[1].lstrip()
			order = 0
			integer = re.findall(r"\d+\.?\d*", words[3])[0]
			try:
				order = int(integer)
			except Exception as e:
				order = int(float(integer) * 10000)
			price = float(re.findall(r"\d+\.?\d*",words[-2])[0])
			text = "课程名称：" + name + " 售价：" + str(price) + " 订阅人数：" + str(order) + "\n"
			
			texts+=text
			if(not names.__contains__(name)):
				names.append(name)
				lesson = Lesson(name,price,order)
				lessons.append(lesson)
	Excel(lessons)
	print(texts)





