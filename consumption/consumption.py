#!/usr/bin/python3

import sys
import argparse
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
import copy
import json

################################################
# General configuration parameters

class Config:
	start_date = None
	end_date = None
	input_file = None
	categories_file = None
	output_dir = None
	year = None
	suffix = None
	colours = None
	ftp = None
	username = None
	password = None

	def __init__(self):
		pass

	def clear(self):
		start_date = None
		end_date = None
		input_file = None
		output_dir = None
		year = None
		latest = None
		suffix = None
		ftp = None
		username = None
		password = None

	# Needed:
	# input csv, input categories, colours, outdir, suffix

	def overlay_json(self, json):
		if 'input' in json:
			self.input_file = json['input']
		if 'categories' in json:
			self.categories_file = json['categories']
		if 'outdir' in json:
			self.output_dir = json['outdir']
		if 'suffix' in json:
			self.suffix = json['suffix']
		if 'colours' in json:
			self.colours = json['colours']
		if 'ftp' in json:
			self.ftp = json['ftp']
		if 'username' in json:
			self.username = json['username']
		if 'password' in json:
			self.password = json['password']

	def overlay_args(self, args):
		if args.input:
			self.input_file = args.input
		if args.categories:
			self.categories_file = args.categories
		if args.outdir:
			self.output_dir = args.outdir
		if args.suffix:
			self.suffix = args.suffix
		if args.colours:
			self.colours = args.colours
		if args.ftp:
			self.ftp = args.ftp
		if args.username:
			self.username = args.username
		if args.password:
			self.password = args.password

	def print_config(self):
		print('Input file: {}'.format(self.input_file))
		print('Categories file: {}'.format(self.categories_file))
		print('Output directory: {}'.format(self.output_dir))
		print('Suffix: {}'.format(self.suffix))
		print('Colours: {}'.format(self.colours))
		print('FTP location: {}'.format(self.ftp))
		print('Username: {}'.format(self.username))
		print('Passsword: {}'.format("Provided" if self.password else 'None'))

################################################
# From graphs.py

class Waste:
	config = Config()
	json = None
	args = None
	dates = None
	types = None
	labels = None
	colours = ['#94070a', '#00381f', '#00864b', '#009353', '#00b274', '#65c295', '#8ccfb7', '#bee3d3']
	file_suffix = '-2022'
	width = None
	upload = None

	categories = ['paper', 'card', 'glass', 'metal', 'returnables', 'compost', 'plastic', 'general']
	categories_reversed = []

	def __init__(self):
		self.categories_reversed = self.categories.copy()
		self.categories_reversed.reverse()
		pass

	@staticmethod
	def get_data_point(types, pos):
		data = []
		for types_pos in range(0, len(types)):
			data.append(types[types_pos][pos])
		return data

	@staticmethod
	def scale_data_point(data, factor):
		for types_pos in range(0, len(data)):
			data[types_pos] = data[types_pos] * factor
		return data

	@staticmethod
	def replace_data_point(types, data, pos):
		for types_pos in range(0, len(types)):
			types[types_pos][pos] = data[types_pos]

	@staticmethod
	def insert_data_point(types, data, pos):
		for types_pos in range(0, len(types)):
			types[types_pos].insert(pos, data[types_pos])

	def load_data(self, input_file):
		with open(input_file, 'rt') as data:
			reader = csv.DictReader(data, delimiter=',', quotechar='"')

			self.dates = []
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
				self.dates.append(datetime.date(2000 + int(datepieces[2]), int(datepieces[1]), int(datepieces[0])))
				paper.append(int(row['Paper']))
				card.append(int(row['Card']))
				glass.append(int(row['Glass']))
				metal.append(int(row['Metal']))
				returnables.append(int(row['Returnables']))
				compost.append(int(row['Compost']))
				plastic.append(int(row['Plastic']))
				general.append(int(row['General']))
				notes.append(row['Notes'])

		self.types = [paper, card, glass, metal, returnables, compost, plastic, general]
		self.types.reverse()

		self.labels = reader.fieldnames[1:-1]
		self.labels.reverse()

	def process_inputs(self):
		print("# Calculated inputs")
		print()

		start_pos = 0
		end_pos = len(self.dates)
		check = 0

		if self.config.year:
			self.config.start_date = datetime.date(int(self.config.year), 1, 1)
			self.config.end_date = datetime.date(int(self.config.year), 12, 31)
			print('Start: {}'.format(self.config.start_date))
			print('End: {}'.format(self.config.end_date))

		if self.config.start_date:
			# Remove entries prior to start_date
			while check < len(self.dates) and self.dates[check] < self.config.start_date:
				check += 1
				start_pos = check

		if self.config.end_date:
			while check < len(self.dates) and self.dates[check] < self.config.end_date:
				check += 1
				end_pos = check

		# Scale the data for the first date if the period overlaps with the start date
		if start_pos > 0:
			factor = 1.0 - ((self.config.start_date - self.dates[start_pos - 1]).days / (self.dates[start_pos] - self.dates[start_pos - 1]).days)
			start_overhang_data = Waste.get_data_point(self.types, start_pos)
			start_overhang_data = Waste.scale_data_point(start_overhang_data, factor)
			Waste.replace_data_point(self.types, start_overhang_data, start_pos)
			# Insert an empty data point at the start
			if factor > 0.0:
				self.dates.insert(start_pos, self.config.start_date)
				empty_data = [0] * len(self.types)
				Waste.insert_data_point(self.types, empty_data, start_pos)
				end_pos += 1

		# Check whether we need to insert an extra date at the end to accommodate overhang
		if end_pos < len(self.dates):
			factor = 1.0 - ((self.dates[end_pos] - self.config.end_date).days / (self.dates[end_pos] - self.dates[end_pos - 1]).days)
			end_overhang_data = Waste.get_data_point(self.types, end_pos)
			end_overhang_data = Waste.scale_data_point(end_overhang_data, factor)
			self.dates.insert(end_pos, self.config.end_date)
			# Insert an extra scaled data point at the end
			Waste.insert_data_point(self.types, end_overhang_data, end_pos)
			end_pos += 1

		self.dates = self.dates[start_pos:end_pos]

		for pos in range(len(self.types)):
			self.types[pos] = self.types[pos][start_pos:end_pos]

		if not self.config.start_date:
			self.config.start_date = self.dates[0]

		if not self.config.end_date:
			self.config.end_date = self.dates[-1]

		# Duration measured in 90-day periods
		self.width = round((self.config.end_date - self.config.start_date).days / 90)
		print('Width: {} periods of 90-days'.format(self.width))

		print('Start date: {}, end date: {}'.format(self.dates[0], self.dates[-1]))

		print()


	daily_pos = 0

	def daily_reset(self):
		self.daily_pos = 0

	def find_interval_start(self, day):
		# The latest entry with date prior to or equal to day
		pos = self.daily_pos
		if (pos < 0) or (pos >= len(self.dates)) or (self.dates[pos] >= day):
			pos = 0
		# Find the date before
		before = pos
		while (pos < len(self.dates)) and (self.dates[pos] <= day):
			if self.dates[before] != self.dates[pos]:
				before = pos
			pos += 1
		self.daily_pos = before
		return before

	def daily_amount(self, day, first_day, last_date, key):
		key_index = self.get_category_index(key)
		accumulation = 0
		pos = self.find_interval_start(day)
		start_pos = pos
		if pos <= 0:
			start_date = first_day
		else:
			start_date = self.dates[pos]

		while (pos <= len(self.dates)) and (self.dates[pos] <= day):
			pos += 1
			accumulation += self.types[key_index][pos]
		if pos == len(self.dates):
			end_date = last_date
		else:
			end_date = self.dates[pos]
		# Number of days for this period
		days = max((end_date - start_date).days, 1)
		# Average quantity per day
		return accumulation / days

	def get_category_index(self, category):
		return self.categories_reversed.index(category)

	def debug_print_amounts(self, start_date, end_date):
		self.daily_reset()
		day = start_date
		while day < end_date:
			general = self.daily_amount(day, start_date, end_date, 'paper')
			plastic = self.daily_amount(day, start_date, end_date, 'plastic')
			print('Date: {}, paper: {}, plastic: {}'.format(day, general, plastic))
			day += datetime.timedelta(days=1)

class Comparison:
	consumption = None
	waste = None
	start = 0
	end = 0
	errors = {}
	file_suffix = '-2022'

	def __init__(self, consumption, waste, start, end):
		self.consumption = consumption
		self.waste = waste
		self.start = start
		self.end = end

	def process_inputs(self):
		for category in self.waste.categories + ['all']:
			self.errors[category] = []

			for offset in range(self.start, self.end):
				error = comparison.average_error(datetime.date(2022, 1, 1), datetime.date(2023, 1, 1), category, offset)
				self.errors[category].append(error)

	def average_error(self, start_date, end_date, category, offset):
		duration = (end_date - start_date).days
		start_consumption = self.consumption.dates[0]
		end_consumption = self.consumption.dates[-1]
		start_waste = self.waste.dates[0]
		end_waste = self.waste.dates[-1]

		print('Calculating average error on overlap')
		print('Requested period: {} -- {}'.format(start_date, end_date))
		print('Requested duration: {} days'.format(duration))
		print('Consumption period: {} -- {}'.format(start_consumption, end_consumption))
		print('Waste period: {} -- {}'.format(start_waste, end_waste))
		print('Offset: {} days'.format(offset))

		start_offset = start_date + datetime.timedelta(days=offset)
		end_offset = end_date + datetime.timedelta(days=offset)

		print('Offset period: {} -- {}'.format(start_offset, end_offset))

		if start_offset < start_waste:
			shift = datetime.timedelta((start_waste - start_offset).days)
			start_offset += shift
			start_date += shift
			print('Updated start date: {}'.format(start_date))

		if end_offset > end_waste:
			shift = (end_offset - end_waste).days
			duration -= shift
			print('Updated duration: {}'.format(duration))

		print()
		print('Constrained period consumption: {} -- {}'.format(start_date, start_date + datetime.timedelta(days=duration)))
		print('Constrained period waste: {} -- {}'.format(start_offset, start_offset + datetime.timedelta(days=duration)))
		print()

		self.consumption.daily_reset()
		self.waste.daily_reset()
		amount_consumption = []
		amount_waste = []
		for pos in range(duration):
			day = start_date + datetime.timedelta(days=pos)
			offset_day = start_date + datetime.timedelta(days=(pos + offset))

			if category == 'all':
				day_consumption = 0
				day_waste = 0
				for check_category in self.waste.categories:
					day_consumption += self.consumption.daily_amount(day, end_date, check_category)
					day_waste += self.waste.daily_amount(offset_day, start_offset, end_offset, check_category)
			else:
				day_consumption = self.consumption.daily_amount(day, end_date, category)
				day_waste = self.waste.daily_amount(offset_day, start_offset, end_offset, category)
			amount_consumption.append(day_consumption)
			amount_waste.append(day_waste)

		scale = sum(amount_consumption)
		if scale != 0:
			amount_consumption = [x / scale for x in amount_consumption]

		scale = sum(amount_waste)
		if scale != 0:
			amount_waste = [x / scale for x in amount_waste]

		error = 0.0
		for pos in range(duration):
			day_consumption = amount_consumption[pos]
			day_waste = amount_waste[pos]
			error += (day_consumption - day_waste)**2
			#print('Date: {}, waste: {}, consumption: {}'.format(day, day_consumption, day_waste))
		average_error = error / duration
		return average_error

	def plot_graphs(self, filecount):
		dpis = [180, 90]
		filenames = ['consumption{:02}{}.png'.format(filecount, self.file_suffix), 'consumption{:02}small{}.png'.format(filecount, self.file_suffix)]
		for filename, dpi in zip(filenames, dpis):
			self.plot_errors(filename, dpi)
		filecount += 1

		waste_categories = self.waste.categories.copy()
		waste_categories.reverse()
		for category in waste_categories + ['all']:
			filenames = ['consumption{:02}{}.png'.format(filecount, self.file_suffix), 'consumption{:02}small{}.png'.format(filecount, self.file_suffix)]
			for filename, dpi in zip(filenames, dpis):
				self.plot_with_offset(category, filename, dpi)
			filecount += 1

	def plot_errors(self, filename, dpi):
		width = 4
		figsize=(width * 2.875 + 0.5, 6)
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)

		waste_categories = self.waste.categories.copy()
		waste_categories.reverse()
		for category in waste_categories + ['all']:
			minimum = min(self.errors[category])
			offset = self.errors[category].index(minimum) + self.start
			print('Category: {}, min: {}, offset: {}'.format(category, minimum, offset))

			consumption.generate_colours(len(waste_categories))

			if category != 'all':
				colour = consumption.colours[waste_categories.index(category)]
			else:
				colour = '#000000'
			ax.plot(range(self.end), self.errors[category], label=category.capitalize(), color=colour)

		ax.set_ylabel("Error $E = \\frac{1}{n}\\sum_{i = 0}^n (\\frac{c_i}{c_T} - \\frac{w_i}{w_T})^2$")
		ax.set_xlabel("Delay between purchase and recycling (days)")
		#ax[0].autoscale(enable=True, axis='x', tight=True)
		ax.set_ylim(bottom=0, top=0.0002)
		#ax[0].set_xlim(left=dates[0], right=dates[len(dates) - 1])
		#ax[0].set_xlim(self.start_date, self.end_date)

		fig.suptitle("Offset error for {}".format(category))

		plt.legend(loc='right')
		plt.tight_layout(pad=2.0, w_pad=0.5)
		plt.savefig(filename, bbox_inches='tight', transparent=True)
		plt.close()

	def plot_with_offset(self, category, filename, dpi):
		dates = []
		for day in range(365):
			dates.append(datetime.date(2022, 1, 1) + datetime.timedelta(days=day))

		minimum = min(self.errors[category])
		offset = self.errors[category].index(minimum) + self.start
		print('Category: {}, offset: {} days'.format(category, offset))

		width = 4
		figsize=(width * 2.875 + 0.5, 6)
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)

		self.waste.daily_reset()
		consumption.daily_reset()
		consumption_daily = []
		waste_daily = []
		for day in dates:
			if category == 'all':
				consumption_sum = 0
				waste_sum = 0
				for category_sum in self.waste.categories:
					consumption_sum += consumption.daily_amount(day, dates[-1], category_sum)
					waste_sum += self.waste.daily_amount(day, dates[0], dates[-1], category_sum)
				consumption_daily.append(consumption_sum)
				waste_daily.append(waste_sum)
			else:
				consumption_daily.append(consumption.daily_amount(day, dates[-1], category))
				waste_daily.append(self.waste.daily_amount(day, dates[0], dates[-1], category))

		scale = sum(consumption_daily)
		if scale != 0:
			consumption_daily = [x / scale for x in consumption_daily]

		scale = sum(waste_daily)
		if scale != 0:
			waste_daily = [x / scale for x in waste_daily]

		ax.plot(dates, consumption_daily, label='Consumption', color='#6e94fc')

		dates_offset = [x + datetime.timedelta(days=offset) for x in dates]
		if offset == 0:
			label='Waste (no offset)'
		else:
			label='Waste ({} day offset)'.format(offset)

		ax.plot(dates_offset, waste_daily, label=label, color='#94070a')

		ax.set_ylabel("Quantity (normalised)")
		ax.set_xlabel("Date")
		#ax[0].autoscale(enable=True, axis='x', tight=True)
		#ax[0].set_ylim(bottom=0, top=ylimit)
		#ax[0].set_xlim(left=dates[0], right=dates[len(dates) - 1])
		#ax[0].set_xlim(self.start_date, self.end_date)

		fig.suptitle("Waste and consumption for {}".format(category))
		plt.legend(loc='right')

		plt.tight_layout(pad=2.0, w_pad=0.5)
		plt.savefig(filename, bbox_inches='tight', transparent=True)
		plt.close()

	def output_table(self):
		print('<table align="center" border="1" cellpadding="4" cellspacing="0">')
		print('\t<tbody>')
		print('\t\t<tr align="left">')
		print('\t\t\t<td>Category</td>')
		print('\t\t\t<td>Min mean square error</td>')
		print('\t\t\t<td>Offset (days)</td>')
		print('\t\t</tr>')
		waste_categories = self.waste.categories.copy()
		waste_categories.reverse()
		for category in waste_categories + ['all']:
			minimum = min(self.errors[category])
			offset = self.errors[category].index(minimum) + self.start
			print('\t\t<tr>')
			print('\t\t\t<td>{}</td>'.format(category))
			print('\t\t\t<td align="right">{:0.3e}</td>'.format(minimum))
			print('\t\t\t<td align="right">{}</td>'.format(offset))
			print('\t\t</tr>')
		print('\t</tbody>')
		print('</table>')


################################################
# Main data and graph management

class Consumption:
	config = Config()
	json = None
	args = None
	dates = None
	types = None
	labels = None
	colours = ['#94070a', '#00381f', '#00864b', '#009353', '#00b274', '#65c295', '#8ccfb7', '#bee3d3']
	file_suffix = '-2022'
	width = None
	upload = None
	categories = None
	purchases = None
	duration = 0
	daily_pos = 0

	def __init__(self):
		pass

	def parse_arguments(self):
		parser = argparse.ArgumentParser(description='Generate graphs about consumption input')
		parser.add_argument("--config", help='read config values from file')
		parser.add_argument("--input", help='CSV data file to load', default='purchases.csv')
		parser.add_argument("--categories", help='CSV categories file to load', default='categories.csv')
		parser.add_argument("--outdir", help='directory to store the generated PNG graphs')
		parser.add_argument("--suffix", help='filename suffix to add; this will be chosen based on the input values if not explicitly provided')
		parser.add_argument("--colours", help='colour scheme to use', choices=['waste', 'consumption'])
		parser.add_argument("--ftp", help="location to use for FTP upload in the form: server/path")
		parser.add_argument("--username", help="username to use for FTP  upload")
		parser.add_argument("--password", help="password to use for FTP upload")
		self.args = parser.parse_args()
		self.config.overlay_args(self.args)

	@staticmethod
	def get_data_point(types, pos):
		data = []
		for types_pos in range(0, len(types)):
			data.append(types[types_pos][pos])
		return data

	@staticmethod
	def scale_data_point(data, factor):
		for types_pos in range(0, len(data)):
			data[types_pos] = data[types_pos] * factor
		return data

	@staticmethod
	def replace_data_point(types, data, pos):
		for types_pos in range(0, len(types)):
			types[types_pos][pos] = data[types_pos]

	@staticmethod
	def insert_data_point(types, data, pos):
		for types_pos in range(0, len(types)):
			types[types_pos].insert(pos, data[types_pos])

	def load_category_data(self, category_file, purchase_file):
		print('Loading category data from: {}'.format(category_file))
		print('Loading purchase data from: {}'.format(purchase_file))

		# Input the categories from the categories file
		self.categories = []
		fields = ['Product', 'Category']
		products = {}
		with open(category_file) as data:
			reader = csv.DictReader(data, delimiter=',', quotechar='"')
			for row in reader:
				category = row['Category']
				product = row['Product']
				if product in products:
					if category != products[product]:
						print('Conflicting product category assignment')
					else:
						print('Repeated product category assignment')
				else:
					if row['Category'] not in self.categories:
						self.categories.append(row['Category'])
					products[product] = category

		product_num = len(products)
		with open(purchase_file, 'rt') as data:
			reader = csv.DictReader(data, delimiter=',', quotechar='"')

			self.dates = []
			self.purchases = []

			current_date = None
			product = None
			gap = 1
			for row in reader:
				if row['Date']:
					datepieces = row['Date'].split("/")
					last_date = current_date
					current_date = datetime.date(2000 + int(datepieces[2]), int(datepieces[1]), int(datepieces[0]))
					if last_date == None:
						last_date = current_date - timedelta(days=gap)
					gap = (current_date - last_date).days
					if gap > 365:
						print('Warning: dates too far apart: {}, {}'.format(current_date, last_date))
					self.dates.append(current_date)
				product = row['Product'].lower()
				category = None
				if product in products:
					category = products[product].lower()
				else:
					print()
					print('"{}": no known category'.format(product))
					for count in range(len(self.categories)):
						print('{:3}: {}'.format(count, self.categories[count]))

				selection = None
				while selection == None and category == None: 
					user_selection = input("Please select a category or enter the name of a new one: ")
					if user_selection.isdigit():
						selection = int(user_selection)
						if selection >= 0 and selection < len(self.categories):
							category = self.categories[selection]
						else:
							selection = None
					else:
						selection = user_selection.lower()
						category = selection
						self.categories.append(category)
				if selection == 'quit':
					break

				#print('{}, {}, {}'.format(current_date, product, category))
				products[product] = category
				item = {}
				item['date'] = current_date
				item['product'] = product
				item['category'] = category
				item['period'] = gap
				item['quantity'] = int(row['Quantity'])
				item['weight'] = float(row['Quantity (g)'])
				item['price'] = float(row['Price (€)']) if row['Price (€)'] else 0.0
				self.purchases.append(item)

#				paper.append(int(row['Paper']))
#				card.append(int(row['Card']))
#				glass.append(int(row['Glass']))
#				metal.append(int(row['Metal']))
#				returnables.append(int(row['Returnables']))
#				compost.append(int(row['Compost']))
#				plastic.append(int(row['Plastic']))
#				general.append(int(row['General']))
#				notes.append(row['Notes'])

#		self.types = [paper, card, glass, metal, returnables, compost, plastic, general]
#		self.types.reverse()

		self.labels = reader.fieldnames[1:-1]
		self.labels.reverse()

		self.duration = (self.dates[-1] - self.dates[0]).days
		print('Total duration: {} days'.format(self.duration))

		product_new = len(products) - product_num
		if product_new > 0:
			print('Writing out products: {} new added'.format(product_new))
			# Output any new categories to the categories file
			with open(category_file, 'w', newline='') as data:
				writer = csv.DictWriter(data, fieldnames = fields, quoting=csv.QUOTE_NONNUMERIC)
				writer.writeheader()
				for product, category in products.items():
					writer.writerow({'Product' : product, 'Category' : category })
		else:
			print('No new categories added')

		self.generate_colours(len(self.categories))

		#self.colours = ['#94070a', '#00381f', '#00864b', '#009353', '#00b274', '#65c295', '#8ccfb7', '#bee3d3']

	def waste_colours(self):
		self.colours = ['#94070a', '#00381f', '#00864b', '#009353', '#00b274', '#65c295', '#8ccfb7', '#bee3d3']

	def generate_colours(self, count):
		self.colours = []		
		step = 2.0 * math.pi / count
		offset = 2.0 * math.pi / 3.0
		for count in range(count):
			self.colours.append(((2.0 + (math.cos((0.0 * offset) + (step * count)))) / 3.0, (2.0 + (math.cos((1.0 * offset) + (step * count)))) / 3.0, (2.0 + (math.cos((2.0 * offset) + (step * count)))) / 3.0))

		for count in range(count - 1):
			if count % 2 == 0:
				self.colours[count], self.colours[count + 1] = self.colours[count + 1], self.colours[count]
		self.colours.reverse()

	def plot_sub_stacked(self, ax, daily):
		bottom = [0]
		for i in range(1, len(daily)):
			bottom.append(bottom[i - 1] + daily[i - 1])

		ax.bar([0], daily, bottom=bottom, color=self.colours, width=1.0)
		ax.set_ylim(ax.get_ylim())
		ax.set_xticklabels([])

	def annual_stats_add_text(self, axis, daily, template, threshold):
		position = 0
		for count in range(len(self.categories)):
			amount = daily[count]
			if amount >= 2.5 * threshold:
				offset = (daily[count] / 2.0) - (threshold / 2.0)
				axis.text(0, position + offset, template.format(amount), ha='center', va='bottom', fontsize=8)
			position += daily[count]

	def annual_stats(self, filenames):
		totals = {'total': {'quantity': 0, 'weight': 0.0, 'price': 0.0}}
		for purchase in self.purchases:
			category = purchase['category']
			item = {}
			if category in totals:
				item = totals[purchase['category']]
				item['quantity'] += purchase['quantity']
				item['weight'] += purchase['weight']
				item['price'] += purchase['price']
			else:
				item['quantity'] = purchase['quantity']
				item['weight'] = purchase['weight']
				item['price'] = purchase['price']
			totals[category] = item
			total = totals['total']
			total['quantity'] += purchase['quantity']
			total['weight'] += purchase['weight']
			total['price'] += purchase['price']
			totals['total'] = total

		print()
		for category, item in totals.items():
			print('Category: {}'.format(category))
			print('Average number per day: {:.3f} items/day'.format(item['quantity'] / self.duration))
			print('Average weight per day: {:.3f} g/day'.format(item['weight'] / self.duration))
			print('Average price per day: {:.2f} €/day'.format(item['price'] / self.duration))
			print('Average weight per item: {:.3f} g/item'.format(item['weight'] / item['quantity']))
			print('Average price per item: {:.2f} €/item'.format(item['price'] / item['quantity']))
			print('Average price per weight: {:.2f} €/kg'.format(1000.0 * item['price'] / item['weight']))
			print()


		dpis = [180, 90]
		for filename, dpi in zip(filenames, dpis):
			print("Generating graph '{}' at {} dpi".format(filename, dpi))
			fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6, 12), dpi=dpi)

			# Average quantities
			daily = [ totals[category]['quantity'] / self.duration for category in self.categories ]
			self.plot_sub_stacked(ax[0], daily)
			ax[0].set_xlabel("Daily number\nitems/day")
			daily_total = total['quantity'] / self.duration
			ax[0].text(0, daily_total, '{:.1f} items'.format(daily_total), ha='center', va='bottom')

			#self.annual_stats_add_text(ax[0], daily, '{:.1f} items', 0.07)

			# Average weights
			daily = [ totals[category]['weight'] / self.duration for category in self.categories ]
			self.plot_sub_stacked(ax[1], daily)
			ax[1].set_xlabel("Daily weight\ng/day")
			daily_total = total['weight'] / self.duration
			ax[1].text(0, daily_total, '{:.0f} g'.format(daily_total), ha='center', va='bottom')

			#self.annual_stats_add_text(ax[1], daily, '{:.0f} g', 12)

			# Average prices
			daily = [ totals[category]['price'] / self.duration for category in self.categories ]
			self.plot_sub_stacked(ax[2], daily)
			ax[2].set_xlabel("Daily price\n€/day")
			daily_total = total['price'] / self.duration
			ax[2].text(0, daily_total, '{:.2f} €'.format(daily_total), ha='center', va='bottom')

			#self.annual_stats_add_text(ax[2], daily, '{:.2f} €', 0.15)

			patches = []
			for count in range(len(self.categories)):
				patches.append(mpatches.Patch(color=self.colours[count], label=self.categories[count].capitalize()))
			patches.reverse()
			fig.legend(handles=patches, loc='right', ncol=1, bbox_to_anchor=(1.3, 0.5))

			fig.suptitle("Daily consumption based on purchases")
			fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
			plt.tight_layout(pad=2.0, w_pad=0.5)
			plt.savefig(filename, bbox_inches='tight', transparent=True)
			plt.close()

	def draw_year_graph(self, value, filenames, title, units):
		start_date = self.purchases[0]['date']
		end_date = self.purchases[-1]['date']
		quantise = 7.0
		types = []
		for category in range(len(self.categories)):
			types.append([])
		dates = []
		for item in self.purchases:
			category_index = self.categories.index(item['category'])
			closest_day = round((item['date'] - start_date).days / quantise) * quantise
			closest_date = start_date + timedelta(days=closest_day)

			if closest_date not in dates:
				types_index = len(dates)
				dates.append(closest_date)
				for count in range(len(self.categories)):
					types[count].append(0.0)
			else:
				types_index = dates.index(closest_date)

			types[category_index][types_index] += item[value]

		dpis = [180, 90]
		width = round((end_date - start_date).days / 90)
		for filename, dpi in zip(filenames, dpis):
			print("Generating graph '{}' at {} dpi".format(filename, dpi))
			graph = Histocurve()
			graph.dates = dates
			graph.types = types
			graph.labels = self.categories
			graph.colours = self.colours
			graph.start_date = start_date
			graph.end_date = end_date
			filepath = self.format_path(filename)
			graph.create_stackedareacurve(width, dpi=dpi, filename=filepath, title=title, units=units)

	def daily_reset(self):
		self.daily_pos = 0

	def find_interval_start(self, day, key):
		# The latest entry with date prior to or equal to day
		pos = self.daily_pos
		if (pos < 0) or (pos >= len(self.purchases)) or (self.purchases[pos]['date'] >= day):
			pos = 0
		# Find the date before
		before = pos
		while (pos < len(self.purchases)) and ((self.purchases[pos]['date'] < day) or (self.purchases[pos]['category'] != key)):
			if (self.purchases[before]['date'] != self.purchases[pos]['date']) and (self.purchases[pos]['category'] == key):
				before = pos
			pos += 1
		if pos < len(self.purchases) and self.purchases[pos]['date'] == day:
			before = pos
		self.daily_pos = before - 1
		return before

	def daily_amount(self, day, last_date, key):
		accumulation = 0
		pos = self.find_interval_start(day, key)
		start_date = self.purchases[pos]['date']

		while (pos < len(self.purchases)) and ((self.purchases[pos]['date'] <= day) or (self.purchases[pos]['category'] != key)):
			if self.purchases[pos]['category'] == key:
				accumulation += self.purchases[pos]['weight']
			pos += 1
		if pos == len(self.purchases):
			end_date = last_date
		else:
			end_date = self.purchases[pos]['date']
		# Number of days for this period
		days = max((end_date - start_date).days, 1)
		# Average quantity per day
		return accumulation / days

	def debug_print_amounts(self, start_date, end_date):
		self.daily_reset()
		day = start_date
		while day < end_date:
			paper = self.daily_amount(day, end_date, 'paper')
			plastic = self.daily_amount(day, end_date, 'plastic')
			print('Date: {}, paper: {}, plastic: {}'.format(day, paper, plastic))
			day += datetime.timedelta(days=1)

	def overview(self):
		if (len(self.dates) > 2):
			print("# Overview")
			print()

	def format_path(self, filename):
		return '{}/{}'.format(self.config.output_dir, filename) if self.config.output_dir else filename

	def plot_graphs(self, filecount=1):
		print("# Plotting data")
		print()

		self.upload = []

		filenames = ['consumption{:02}{}.png'.format(filecount, self.file_suffix), 'consumption{:02}small{}.png'.format(filecount, self.file_suffix)]
		self.upload = self.upload + filenames
		self.annual_stats(filenames)
		filecount += 1

		filenames = ['consumption{:02}{}.png'.format(filecount, self.file_suffix), 'consumption{:02}small{}.png'.format(filecount, self.file_suffix)]
		self.upload = self.upload + filenames
		self.draw_year_graph('weight', filenames, 'Weight of household purchases', 'g')
		filecount += 1

		filenames = ['consumption{:02}{}.png'.format(filecount, self.file_suffix), 'consumption{:02}small{}.png'.format(filecount, self.file_suffix)]
		self.upload = self.upload + filenames
		self.draw_year_graph('price', filenames, 'Cost of household purchases', '€')
		filecount += 1

		print()

	def ftp_upload(self):
		# Upload the result
		if self.config.ftp:
			print("# Uploading data")
			print()

			ftp_server = ''
			ftp_path = ''
			if self.config.ftp:
				transport = self.config.ftp.find(':')
				transport = transport + 3 if transport >= 0 else 0
				path = self.config.ftp[transport:].find('/')
				if path >= 0:
					ftp_server = self.config.ftp[:transport + path]
					ftp_path = self.config.ftp[transport + path:]
				else:
					ftp_server = self.config.ftp
					ftp_path = '/'
				print('FTP server: {}'.format(ftp_server))
				print('FTP path: {}'.format(ftp_path))

			print
			username = self.config.username
			password = self.config.password
			if not username or not password:
				print('Please authenticate to {}'.format(ftp_server))
				if username:
					print('Username: {}'.format(username))
				else:
					username = input("Username: ")
				if password:
					print('Password: {}'.format('*****'))
				else:
					password = getpass.getpass()

			print("Logging in to {} as {}".format(ftp_server, username))
			ftp = FTP(ftp_server)
			ftp.login(username, password)
			ftp.cwd(ftp_path)

			print("Uploading files")
			for filename in self.upload:
				print("Uploading '{}'".format(filename))
				filepath = self.format_path(filename)
				ftp.storbinary('STOR {}'.format(filename), open(filepath, 'rb'))

			ftp.quit()
			print()

	def draw(self, count = None):
		print("# Input parameters")
		print()
		if count != None:
			print('Repeat: {}'.format(count))
		print('Config: {}'.format(self.args.config))
		self.config.print_config()
		print()
		self.load_data(self.config.input_file)
		self.process_inputs()
		self.overview()


		self.load_category_data(self.config.categories_file, self.config.input_file)
		self.plot_graphs()
		#self.ftp_upload()

	def execute_config(self):
		json_data = {}
		if self.args.config:
			with open(self.args.config, 'rt') as config_json:
				json_data = json.loads(config_json.read())

		if 'repeat' in json_data:
			count = 0
			for step in json_data['repeat']:
				self.config.clear()
				self.config.overlay_json(json_data)
				self.config.overlay_json(step)
				self.config.overlay_args(graphs.args)
				self.draw(count)
				count += 1
		else:
			self.config.clear()
			self.config.overlay_json(json_data)
			self.config.overlay_args(graphs.args)
			self.draw()

################################################
# Stacked line graph with average

class LineGraph:
	dates = None
	types = None
	labels = None
	colours = None
	start_date = None
	end_date = None

	def __init__(self):
		pass

	def create_plot(self, width, dpi, filename):
		figsize = (width * 2.875 + 0.5, 6)
		ratios = [width * 2.875, 0.5]
		fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': ratios}, figsize=figsize, dpi=dpi)
		y = np.vstack(self.types)
		ax[0].stackplot(self.dates[0:], y, labels=self.labels, colors=self.colours)

		handles, legend_labels = ax[0].get_legend_handles_labels()
		handles.reverse()
		legend_labels.reverse()
		ax[0].legend(handles, legend_labels, loc='upper left')

		ax[0].set_ylabel("Quantity disposed (g)")
		ax[0].set_xlabel("Date")
		#ax[0].autoscale(enable=True, axis='x', tight=True)
		ax[0].set_xlim(self.start_date, self.end_date)

		# Averages
		duration = self.dates[len(self.dates) - 1].toordinal() - self.dates[0].toordinal()
		sums = [sum(items) for items in self.types]
		averages = [ x / float(duration) for x in sums ]
		weekly = [ x * 7.0 for x in averages ]

		bottom = [0]
		for i in range(1, len(weekly)):
			bottom.append(bottom[i - 1] + weekly[i - 1])

		ax[1].bar([0], weekly, bottom=bottom, color=self.colours, width=1.0)

		ax[1].set_ylim(ax[0].get_ylim())

		ax[1].set_xlabel("Weekly\naverage")
		ax[1].yaxis.set_label_position("right")
		ax[1].yaxis.tick_right()
		ax[1].set_xticklabels([])

		fig.suptitle("Quantity of waste and recycling")
		fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
		plt.tight_layout(pad=2.0, w_pad=0.5)
		plt.savefig(filename, bbox_inches='tight', transparent=True)
		plt.close()

################################################
# Stacked histocurve with average

class Histocurve:
	dates = None
	types = None
	labels = None
	colours = None
	start_date = None
	end_date = None

	def __init__(self):
		pass

	@staticmethod
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

	@staticmethod
	def ydash(x, y, i):
		if i < 1:
			return Histocurve.ydash(x, y, 1)
		elif i >= len(x):
			return Histocurve.ydash(x, y, len(x) - 1)
		else:
			return y[i] / (x[i] - x[i - 1])

	@staticmethod
	def Xval(x, y, i):
		if i < 0:
			return x[0]
		elif i >= 2 * len(x):
			return x[len(x) - 1]
		elif i % 2 == 0:
			return x[(int)(i / 2)]
		else:
			return 0.5 * (Histocurve.Xval(x, y, i + 1) + Histocurve.Xval(x, y, i - 1))

	@staticmethod
	def Yval(x, y, i):
		if i == 0:
			return Histocurve.ydash(x, y, 1)
		elif i % 2 == 0:
			return 0.5 * (Histocurve.ydash(x, y, (int)(i / 2)) + Histocurve.ydash(x, y, (int)(i / 2) + 1))
		else:
			return Histocurve.ydash(x, y, (int)((i + 1) / 2)) + ((Histocurve.ydash(x, y, (int)((i + 1) / 2)) - Histocurve.Yval(x, y, i - 1)) / 2.0) + ((Histocurve.ydash(x, y, (int)((i + 1) / 2)) - Histocurve.Yval(x, y, i + 1)) / 2.0)

	@staticmethod
	def generate_areacurve(x, y):
		X = []
		Y = []

		points = len(x) * 2

		for i in range(0, points):
			X.append(Histocurve.Xval(x, y, i))
			Y.append(Histocurve.Yval(x, y, i))

		handles = []
		handles.append((X[0], Y[0]))

		for i in range(0, points - 2):
			gradient_left, gradient_right, xwidth, area1 = Histocurve.gradients(X, Y, i)

			Yleft = Y[i]
			Yright = Y[i + 1]

			if i % 2 == 0:
				# Adjust to compensate for change in volume
				gradient_left2, gradient_right2, xwidth2, area2 = Histocurve.gradients(X, Y, i + 1)
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

	@staticmethod
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

	@staticmethod
	def stackcurves(top, bottom):
		if len(top) != len(bottom):
			print("Stacked curves must have same number of points {} != {}".format(len(top), len(bottom)))
			error("Stacked curves must have same number of points")
		raised = []
		for i in range(0, len(top)):
			point = [top[i][0], top[i][1] + bottom[i][1]]
			raised.append(point)
		return raised

	def create_stackedareacurve(self, width, dpi, filename, title, units):
		figsize=(width * 2.875 + 0.5, 6)
		ratios = [width * 2.875, 0.5]
		fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': ratios}, figsize=figsize, dpi=dpi)
		x = [date.toordinal() for date in self.dates]

		ground = [0] * (len(self.types[0]) + 1)
		datas = copy.deepcopy(self.types)

		bottom = Histocurve.generate_areacurve(x, ground)
		patches = []
		ymax = 0.0
		for i in range(0, len(datas)):
			top = Histocurve.generate_areacurve(x, datas[i])
			top2 = Histocurve.stackcurves(top, bottom)
			ymax = max([ymax] + [p[1] for p in top2])

			handles, codes = Histocurve.bound_areacurve(top2, bottom)

			path = mpatches.Path(handles, codes)

			patches.append(mpatches.PathPatch(path, color="None", fc=self.colours[i], transform=ax[0].transData))

			bottom = top2

		patches.reverse()
		for patch in patches:
			ax[0].add_patch(patch)

		ymax *= 1.03
		ymax = math.ceil(ymax / 20.0) * 20.0

		#ax[0].plot(X, Y)

		legend_labels = self.labels[:]
		legend_labels.reverse()
		ax[0].legend(legend_labels, loc='upper left')

		ax[0].set_ylabel("Quantity purchased ({} / day)".format(units))
		ax[0].set_xlabel("Date")
		#ax[0].autoscale(enable=True, axis='x', tight=True)
		ax[0].set_ylim(bottom=0, top=ymax)
		#ax[0].set_xlim(left=dates[0], right=dates[len(dates) - 1])
		ax[0].set_xlim(self.start_date.toordinal(), self.end_date.toordinal())

		# Display the x axis labels as dates rather than numbers
		def formatter(x, pos):
			return datetime.date.fromordinal(int(x))
		ax[0].xaxis.set_major_formatter(formatter)

		# Averages
		duration = self.dates[len(self.dates) - 1].toordinal() - self.dates[0].toordinal()
		sums = [sum(items) for items in self.types]
		averages = [ x / float(duration) for x in sums ]
		daily = [ x * 1.0 for x in averages ]

		bottom = [0]
		for i in range(1, len(daily)):
			bottom.append(bottom[i - 1] + daily[i - 1])

		ax[1].bar([0], daily, bottom=bottom, color=self.colours, width=1.0)

		ax[1].set_ylim(ax[0].get_ylim())

		ax[1].set_xlabel("Daily\npurchased")
		ax[1].yaxis.set_label_position("right")
		ax[1].yaxis.tick_right()
		ax[1].set_xticklabels([])

		fig.suptitle(title)
		fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
		plt.tight_layout(pad=2.0, w_pad=0.5)
		plt.savefig(filename, bbox_inches='tight', transparent=True)
		plt.close()

################################################
# Histogram with average

class Histogram:
	dates = None
	start_date = None
	end_date = None

	def __init__(self):
		pass

	def create_histogram(self, width, dpi, filename, data, ylimit, colour, title):
		figsize=(width * 2.875 + 0.5, 6)
		ratios = [width * 2.875, 0.5]
		fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': ratios}, figsize=figsize, dpi=dpi)

		ymax = 0.0

		widths = [-7]
		for i in range(1, len(self.dates)):
			widths.append(self.dates[i - 1].toordinal() - self.dates[i].toordinal())

		y = data[:]
		#y.insert(0, y[0])
		for i in range(0, len(widths)):
			y[i] = y[i] / -widths[i]

		ymax = max(y)
		ymax *= 1.03
		ymax = math.ceil(ymax / 50.0) * 50.0

		if ylimit == 0:
			ylimit = ymax

		ax[0].bar(self.dates, y, widths, align='edge', color=colour, edgecolor='black')

		ax[0].set_ylabel("Quantity disposed (g / day)")
		ax[0].set_xlabel("Date")
		#ax[0].autoscale(enable=True, axis='x', tight=True)
		ax[0].set_ylim(bottom=0, top=ylimit)
		#ax[0].set_xlim(left=dates[0], right=dates[len(dates) - 1])
		ax[0].set_xlim(self.start_date, self.end_date)

		# Averages
		duration = self.dates[len(self.dates) - 1].toordinal() - self.dates[0].toordinal()
		sums = sum(data)

		average = sum(data) / float(duration)

		ax[1].bar([0], average, color=colour, width=1.0)

		ax[1].set_ylim(ax[0].get_ylim())

		ax[1].set_xlabel("Daily\naverage")
		ax[1].yaxis.set_label_position("right")
		ax[1].yaxis.tick_right()
		ax[1].set_xticklabels([])

		if title != '':
			fig.suptitle("Quantity of waste and recycling - " + title)
		else:
			fig.suptitle("Quantity of waste and recycling")

		plt.tight_layout(pad=2.0, w_pad=0.5)
		plt.savefig(filename, bbox_inches='tight', transparent=True)
		plt.close()


################################################
# Main

# Data file format
#
# CSV file with a headings line followed by data lines
# Example:
#
# Date,Paper,Card,Glass,Metal,Returnables,Compost,Plastic,General,Notes
# 11/08/19,0,0,0,0,0,0,0,0,"Not a reading"
# 18/08/19,221,208,534,28,114,584,0,426,"A proper reading"

# Config file format
#
# JSON file containing config values
# If a repeat array is included then multiple runs will be performed
# Example:
#
# {
# 	"start": "<YYYY-MM-DD>",
# 	"end": "<YYYY-MM-DD>",
# 	"input": "<filename>",
# 	"year": <year>,
# 	"latest": <0|1>,
# 	"suffix": "<suffix>",
# 	"ftp": "<server>/<path>",
# 	"username": "<username>",
# 	"password": "<password>",
#
# 	"repeat": [
# 		{
# 			"start": "<YYYY-MM-DD>",
# 			...
# 		}
# 	]
# }

consumption = Consumption()
consumption.parse_arguments()
#consumption.execute_config() # Should be used instead of draw()
consumption.load_category_data('categories-min.csv', 'purchases.csv')
#consumption.annual_stats(['consumption01'])
#consumption.draw() # Should be used instead of plot_graphs()
consumption.plot_graphs(1)
#consumption.debug_print_amounts(datetime.date(2022, 1, 1), datetime.date(2022, 12, 31))

consumption = Consumption()
consumption.parse_arguments()
#consumption.execute_config() # Should be used instead of draw()
consumption.load_category_data('categories-waste.csv', 'purchases.csv')
consumption.waste_colours()

#consumption.annual_stats(['consumption01'])
#consumption.draw() # Should be used instead of plot_graphs()
consumption.plot_graphs(4)
#consumption.debug_print_amounts(datetime.date(2022, 1, 1), datetime.date(2022, 12, 31))

#exit()

waste = Waste()
waste.load_data('../recycling.csv')
waste.process_inputs()

#waste.debug_print_amounts(datetime.date(2022, 1, 1), datetime.date(2022, 12, 31))

comparison = Comparison(consumption, waste, 0, 91)
comparison.process_inputs()
comparison.plot_graphs(7)
comparison.output_table()





print("All done")


