import xml.etree.ElementTree as et
import os
import xml.dom.minidom



DIR_TEST = '../test_data/Images'
dir_path = '../test_predictions/XMLFiles'

def generatexmlfile(filename,imgname,image,draw_boxes,pred_classes):

	save_path = os.path.join(dir_path, filename)
	root = et.Element("annotation")
	s1 = et.Element("folder")
	s1.text = DIR_TEST
	root.append(s1)

	s2 = et.Element("filename")
	s2.text = imgname
	root.append(s2)

	path = et.Element('path')
	path.text = os.path.join(DIR_TEST,imgname)
	root.append(path)

	d1 = et.Element("source")
	s3 = et.SubElement(d1,"database")
	s3.text = 'unknown'
	root.append(d1) 


	image_height , image_width , depth = image.shape

	d2 = et.Element('size')
	s4 = et.SubElement(d2,'width') 
	s4.text = str(image_width)
	s5 = et.SubElement(d2,'height')
	s5.text = str(image_height)
	s6 = et.SubElement(d2,'depth')
	s6.text = str(depth)

	root.append(d2)

	s7 = et.Element("segmented")
	s7.text = str(0)
	root.append(s7)
	print(draw_boxes)
	print(pred_classes)
	for i , boxes in enumerate(draw_boxes):
		d3 = et.Element('object')

		name = et.SubElement(d3,'name')
		name.text = pred_classes[i]

		pose = et.SubElement(d3,'pose')
		pose.text = "Unspecified"

		truncated = et.SubElement(d3,'truncated')
		truncated.text = str(0)

		difficult = et.SubElement(d3,'difficult')
		difficult.text = str(0)

		occluded = et.SubElement(d3,'occluded')
		occluded.text = str(0)


		d4 = et.Element('bndbox')

		xmin = et.SubElement(d4,'xmin')
		xmin.text = str(boxes[0])

		ymin = et.SubElement(d4,'ymin')
		ymin.text = str(boxes[1])

		xmax = et.SubElement(d4,'xmax')
		xmax.text = str(boxes[2])

		ymax = et.SubElement(d4,'ymax')
		ymax.text = str(boxes[3])

		d3.append(d4)
		root.append(d3)

	tree = et.ElementTree(root)
	et.indent(tree)
	tree.write(save_path)




