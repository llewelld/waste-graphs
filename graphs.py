#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import csv
import time
from datetime import timedelta
import datetime
import getpass
import math
from ftplib import FTP

################################################
# Data loading

with open('recycling.csv', 'rt') as data:
	reader = csv.DictReader(data, delimiter=',', quotechar='"')

	dates = []
	paper = []
	card = []
	glass = []
	metal = []
	returnables = []
	compost = []
	plastic = []
	general = []
	notes = []
	
	#Date,Paper,Card,Glass,Metal,Returnables/cans,Compost,Plastic,General
	
	for row in reader:
		datepieces = row['Date'].split("/")
		dates.append(datetime.date(2000 + int(datepieces[2]), int(datepieces[1]), int(datepieces[0])))
		paper.append(int(row['Paper']))
		card.append(int(row['Card']))
		glass.append(int(row['Glass']))
		metal.append(int(row['Metal']))
		returnables.append(int(row['Returnables']))
		compost.append(int(row['Compost']))
		plastic.append(int(row['Plastic']))
		general.append(int(row['General']))
		notes.append(row['Notes'])

types = [paper, card, glass, metal, returnables, compost, plastic, general]
types.reverse()
y = np.vstack(types)
colours=['#94070a', '#00381f', '#00864b', '#009353', '#00b274', '#65c295', '#8ccfb7', '#bee3d3']

labels = reader.fieldnames[1:-1]
labels.reverse()

dates.insert(0, dates[0] - timedelta(days = 7))

################################################
# Original stacked line graph

def create_plot(figsize, dpi, filename):
	fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [20, 1]}, figsize=figsize, dpi=dpi)
	ax[0].stackplot(dates[1:], y, labels=labels, colors=colours)

	handles, legend_labels = ax[0].get_legend_handles_labels()
	handles.reverse()
	legend_labels.reverse()
	ax[0].legend(handles, legend_labels, loc='upper left')

	ax[0].set_ylabel("Quantity disposed (g)")
	ax[0].set_xlabel("Date")
	ax[0].autoscale(enable=True, axis='x', tight=True)

	# Averages
	duration = dates[len(dates) - 1].toordinal() - dates[0].toordinal()
	sums = [sum(paper), sum(card), sum(glass), sum(metal), sum(returnables), sum(compost), sum(plastic), sum(general)]
	sums.reverse()
	averages = [ x / float(duration) for x in sums ]
	weekly = [ x * 7.0 for x in averages ]

	bottom = [0]
	for i in range(1, len(weekly)):
		bottom.append(bottom[i - 1] + weekly[i - 1])
		
	ax[1].bar([0], weekly, bottom=bottom, color=colours, width=1.0)

	ax[1].set_ylim(ax[0].get_ylim())

	ax[1].set_xlabel("Weekly\naverage")
	ax[1].yaxis.set_label_position("right")
	ax[1].yaxis.tick_right()
	ax[1].set_xticklabels([])

	fig.suptitle("Quantity of waste and recycling")
	fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
	plt.tight_layout(pad=2.0, w_pad=0.5)
	plt.savefig(filename, bbox_inches='tight', transparent=True)


################################################
# Stacked histocurve

def gradients(X, Y, i):
	points = len(X)

	xwidth = X[i + 1] - X[i]
	area = 0.5 * (xwidth * (Y[i + 1] - Y[i]))
	if area < 0:
		area = -area

	if i > 0:
		gradient_left = (Y[i + 1] - Y[i - 1]) / (X[i + 1] - X[i - 1]) 
	else:
		gradient_left = (Y[i + 1] - Y[i - 0]) / (X[i + 1] - X[i - 0]) 

	if i < points - 2:
		gradient_right = (Y[i + 2] - Y[i - 0]) / (X[i + 2] - X[i - 0]) 
	else:
		gradient_right = (Y[i + 1] - Y[i - 0]) / (X[i + 1] - X[i - 0])

	return gradient_left, gradient_right, xwidth, area

def ydash(x, y, i):
	if i < 1:
		return ydash(x, y, 1)
	elif i >= len(x):
		return ydash(x, y, len(x) - 1)
	else:
		return y[i] / (x[i] - x[i - 1])

def Xval(x, y, i):
	if i < 0:
		return x[0]
	elif i >= 2 * len(x):
		return x[len(x) - 1]
	elif i % 2 == 0:
		return x[(int)(i / 2)]
	else:
		return 0.5 * (Xval(x, y, i + 1) + Xval(x, y, i - 1))

def Yval(x, y, i):
	if i == 0:
		return ydash(x, y, 1)
	elif i % 2 == 0:
		return 0.5 * (ydash(x, y, (int)(i / 2)) + ydash(x, y, (int)(i / 2) + 1))
	else:
		return ydash(x, y, (int)((i + 1) / 2)) + ((ydash(x, y, (int)((i + 1) / 2)) - Yval(x, y, i - 1)) / 2.0) + ((ydash(x, y, (int)((i + 1) / 2)) - Yval(x, y, i + 1)) / 2.0)	

def generate_areacurve(x, y):
	X = []
	Y = []

	points = len(x) * 2

	for i in range(0, points):
		X.append(Xval(x, y, i))
		Y.append(Yval(x, y, i))

	handles = []
	handles.append((X[0], Y[0]))

	for i in range(0, points - 2):
		gradient_left, gradient_right, xwidth, area1 = gradients(X, Y, i)

		Yleft = Y[i]
		Yright = Y[i + 1]

		if i % 2 == 0:
			# Adjust to compensate for change in volume
			gradient_left2, gradient_right2, xwidth2, area2 = gradients(X, Y, i + 1)
			area = area1 + area2
			B = (gradient_left * xwidth / 3.0)
			Cdash = - (gradient_right * xwidth / 3.0)
			Edash = (gradient_left2 * xwidth2 / 3.0)
			F = - (gradient_right2 * xwidth2 / 3.0)
			Ddelta = - 0.5 * (B + Cdash + Edash + F) / (xwidth + xwidth2)
			D = area - 0.25 * (B + Cdash + Edash + F)
			D = Y[i + 1] + Ddelta
			Yright = D

			#ax[0].plot(X[i], Y[i], "ro")
		else:
			Yleft = D

		handles.append((X[i] + xwidth / 3.0, Yleft + (gradient_left * xwidth / 3.0)))
		handles.append((X[i + 1] - xwidth / 3.0, Yright - (gradient_right * xwidth / 3.0)))
		handles.append((X[i + 1], Yright))


	return handles

def bound_areacurve(top, bottom):
	handles = top[:]
	codes = [mpatches.Path.MOVETO] + [mpatches.Path.CURVE4] * (len(top) - 1)
	bottom2 = bottom[:]
	bottom2.reverse()
	handles += bottom2
	codes += [mpatches.Path.LINETO] + [mpatches.Path.CURVE4] * (len(bottom) - 1)
	handles.append(top[0])
	codes.append(mpatches.Path.CLOSEPOLY)
	return handles, codes

def stackcurves(top, bottom):
	if len(top) != len(bottom):
		print("Stacked curves must have same number of points {} != {}".format(len(top), len(bottom)))
		error("Stacked curves must have same number of points")
	raised = []
	for i in range(0, len(top)):
		point = [top[i][0], top[i][1] + bottom[i][1]]
		raised.append(point)
	return raised

def create_stackedareacurve(figsize, dpi, filename):
	fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [20, 1]}, figsize=figsize, dpi=dpi)
	x = [date.toordinal() for date in dates]

	ground = [0] * (len(general) + 1)
	datas = [general[:], plastic[:], compost[:], returnables[:], metal[:], glass[:], card[:], paper[:]]
	for data in datas:
		data.insert(0, data[0])

	colours=['#94070a', '#00381f', '#00864b', '#009353', '#00b274', '#65c295', '#8ccfb7', '#bee3d3']

	bottom = generate_areacurve(x, ground)
	patches = []
	ymax = 0.0
	for i in range(0, len(datas)):
		top = generate_areacurve(x, datas[i])
		top2 = stackcurves(top, bottom)
		ymax = max([ymax] + [p[1] for p in top2])

		handles, codes = bound_areacurve(top2, bottom)

		path = mpatches.Path(handles, codes)

		patches.append(mpatches.PathPatch(path, color="None", fc=colours[i], transform=ax[0].transData))

		bottom = top2

	patches.reverse()
	for patch in patches:
		ax[0].add_patch(patch)

	ymax *= 1.03
	ymax = math.ceil(ymax / 50.0) * 50.0

	#ax[0].plot(X, Y)

	legend_labels = labels[:]
	legend_labels.reverse()
	ax[0].legend(legend_labels, loc='upper left')

	ax[0].set_ylabel("Quantity disposed (g / day)")
	ax[0].set_xlabel("Date")
	ax[0].autoscale(enable=True, axis='x', tight=True)
	ax[0].set_ylim(bottom=0, top=ymax)
	ax[0].set_xlim(left=dates[0], right=dates[len(dates) - 1])


	# Averages
	duration = dates[len(dates) - 1].toordinal() - dates[0].toordinal()
	sums = [sum(paper), sum(card), sum(glass), sum(metal), sum(returnables), sum(compost), sum(plastic), sum(general)]
	sums.reverse()
	averages = [ x / float(duration) for x in sums ]
	daily = [ x * 1.0 for x in averages ]

	bottom = [0]
	for i in range(1, len(daily)):
		bottom.append(bottom[i - 1] + daily[i - 1])
		
	ax[1].bar([0], daily, bottom=bottom, color=colours, width=1.0)

	ax[1].set_ylim(ax[0].get_ylim())

	ax[1].set_xlabel("Daily\naverage")
	ax[1].yaxis.set_label_position("right")
	ax[1].yaxis.tick_right()
	ax[1].set_xticklabels([])
	print ("Daily averages:", daily)

	fig.suptitle("Quantity of waste and recycling")
	fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
	plt.tight_layout(pad=2.0, w_pad=0.5)
	plt.savefig(filename, bbox_inches='tight', transparent=True)


################################################
# Histogram with average

def create_histogram(figsize, dpi, filename, data, ylimit, colour, title):
	fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [20, 1]}, figsize=figsize, dpi=dpi)

	ymax = 0.0

	widths = [-7]
	for i in range(1, len(dates)):
		widths.append(dates[i - 1].toordinal() - dates[i].toordinal())

	y = data[:]
	y.insert(0, y[0])
	for i in range(0, len(widths)):
		y[i] = y[i] / -widths[i]

	ymax = max(y)
	ymax *= 1.03
	ymax = math.ceil(ymax / 50.0) * 50.0

	if ylimit == 0:
		ylimit = ymax

	ax[0].bar(dates, y, widths, align='edge', color=colour, edgecolor='black')

	ax[0].set_ylabel("Quantity disposed (g / day)")
	ax[0].set_xlabel("Date")
	ax[0].autoscale(enable=True, axis='x', tight=True)
	ax[0].set_ylim(bottom=0, top=ylimit)
	ax[0].set_xlim(left=dates[0], right=dates[len(dates) - 1])

	# Averages
	duration = dates[len(dates) - 1].toordinal() - dates[0].toordinal()
	sums = sum(data)

	average = sum(data) / float(duration)

	ax[1].bar([0], average, color=colour, width=1.0)

	ax[1].set_ylim(ax[0].get_ylim())

	ax[1].set_xlabel("Daily\naverage")
	ax[1].yaxis.set_label_position("right")
	ax[1].yaxis.tick_right()
	ax[1].set_xticklabels([])

# em-dash \xe2

	if title != '':
		fig.suptitle("Quantity of waste and recycling - " + title)
	else:
		fig.suptitle("Quantity of waste and recycling")

	plt.tight_layout(pad=2.0, w_pad=0.5)
	plt.savefig(filename, bbox_inches='tight', transparent=True)


################################################
# Main

if (len(dates) > 2):
	print("# Overview")
	print()
	start_date = dates[0]
	penultimate_date = dates[len(dates) - 2]
	end_date = dates[len(dates) - 1]
	duration = (end_date - start_date).days
	total = sum([sum(wastetype) for wastetype in types])
	latest_total = sum([wastetype[len(wastetype) - 1] for wastetype in types])
	latest_duration = (end_date - penultimate_date).days

	start_year = start_date.year
	this_year = end_date.year

	year_average = {}
	for year in range(start_year, this_year + 1):
		year_total = 0
		start = 0
		end = 0
		for i in range(len(dates) - 1):
			if dates[i + 1].year == year:
				if start == 0:
					start = dates[i]
				year_total += sum([wastetype[i] for wastetype in types])
				end = dates[i + 1]

		year_duration = (end - start).days
		if year_duration > 0:
			year_average[year] = year_total / year_duration

	print("Total period: \t{} - {} ({} days)".format(start_date, end_date, duration))
	print("Latest period: \t{} - {} ({} days)".format(penultimate_date, end_date, latest_duration))
	print()

	print("Overall daily average: \t\t{:.2f} g/day".format(total / duration))

	for year in year_average:
		print("Year {} daily average: \t{:.2f} g/day".format(year, year_average[year]))

	print("Latest entry daily average: \t{:.2f} g/day".format(latest_total / latest_duration))
	print()

print("# Plotting data")
print()

upload = []

# Stacked line graph for debugging purposes, so don't upload
filenames = ['waste01.png', 'waste01small.png']
upload = upload + filenames
dpis = [180, 90]
for filename, dpi in zip(filenames, dpis):
	print("Generating graph '{}' at {} dpi".format(filename, dpi))
	create_plot(figsize=(12, 6), dpi=dpi, filename=filename)


filenames = ['waste08.png', 'waste08small.png']
upload = upload + filenames
for filename, dpi in zip(filenames, dpis):
	print("Generating graph '{}' at {} dpi".format(filename, dpi))
	create_stackedareacurve(figsize=(12, 6), dpi=dpi, filename=filename)

for i in range(0, len(types)):
	filenames = []
	filenames.append("waste-detail0{}-{}.png".format(i, labels[i].lower()))
	filenames.append("waste-detail0{}small-{}.png".format(i, labels[i].lower()))
	upload = upload + filenames
	for filename, dpi in zip(filenames, dpis):
		print("Generating graph '{}' at {} dpi".format(filename, dpi))
		create_histogram(figsize=(12, 6), dpi=dpi, filename=filename, data=types[i], ylimit=0, colour=colours[i], title=labels[i])

# Upload the result
location = ''
path = ''
username = ''
print
location_check = input('Server name ({}): '.format(location))
if location_check != '':
	location = location_check

path_check = input('Folder path ({}): '.format(path))
if path_check != '':
	path = path_check

print("Please authenticate to {}".format(location))
username_check = input("Username ({}): ".format(username))
if username_check != '':
	username = username_check
password = getpass.getpass()

print("Logging in to {} as {}".format(location, username))
ftp = FTP(location)
ftp.login(username, password)
ftp.cwd(path)

print("Uploading files")
for filename in upload:
	print("Uploading '{}'".format(filename))
	ftp.storbinary('STOR {}'.format(filename), open(filename, 'rb'))

ftp.quit()
print("All done")

