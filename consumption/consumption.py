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
	output_dir = None
	year = None
	latest = None
	suffix = None
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

	def overlay_json(self, json):
		if 'start' in json:
			self.start_date = json['start']
		if 'end' in json:
			self.end_date = json['end']
		if 'input' in json:
			self.input_file = json['input']
		if 'outdir' in json:
			self.output_dir = json['outdir']
		if 'year' in json:
			self.year = json['year']
		if 'latest' in json:
			self.latest = json['latest']
		if 'suffix' in json:
			self.suffix = json['suffix']
		if 'ftp' in json:
			self.ftp = json['ftp']
		if 'username' in json:
			self.username = json['username']
		if 'password' in json:
			self.password = json['password']
		if 'all' in json:
			self.start_date = None
			self.end_date = None
			self.year = None
			self.latest = None

	def overlay_args(self, args):
		if args.start:
			self.start_date = args.start
		if args.end:
			self.end_date = args.end
		if args.input:
			self.input_file = args.input
		if args.outdir:
			self.output_dir = args.outdir
		if args.year:
			self.year = args.year
		if args.latest:
			self.latest = args.latest
		if args.suffix:
			self.suffix = args.suffix
		if args.ftp:
			self.ftp = args.ftp
		if args.username:
			self.username = args.username
		if args.password:
			self.password = args.password
		if args.all:
			self.start_date = None
			self.end_date = None
			self.year = None
			self.latest = None

	def print_config(self):
		print('Start: {}'.format(self.start_date))
		print('End: {}'.format(self.end_date))
		print('Input file: {}'.format(self.input_file))
		print('Output directory: {}'.format(self.output_dir))
		print('Year: {}'.format(self.year))
		print('Latest: {}'.format(self.latest))
		print('Suffix: {}'.format(self.suffix))
		print('FTP location: {}'.format(self.ftp))
		print('Username: {}'.format(self.username))
		print('Passsword: {}'.format("Provided" if self.password else 'None'))

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
	file_suffix = ''
	width = None
	upload = None
	categories = None
	purchases = None
	duration = 0

	def __init__(self):
		pass

	def read_date(string):
		return datetime.date.fromisoformat(string)

	def parse_arguments(self):
		parser = argparse.ArgumentParser(description='Generate graphs about waste output')
		parser.add_argument("--config", help='read config values from file')
		parser.add_argument("--start", type=self.read_date, help='start date to plot from')
		parser.add_argument("--end", type=self.read_date, help='end date to plot to')
		parser.add_argument("--input", help='CSV data file to load', default='recycling.csv')
		parser.add_argument("--outdir", help='Directory to store the generated PNG graphs')
		parser.add_argument("--year", type=int, help='plot graphs for a given year; overrides start and end arguments')
		parser.add_argument("--latest", help='plot the most recent year; overrides start, end and year arguments', action="store_true")
		parser.add_argument("--all", help='plot all values; overrides other time values', action="store_true")
		parser.add_argument("--suffix", help='filename suffix to add; this will be chosen based on the input values if not explicitly provided')
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

		self.colours = []		
		step = 2.0 * math.pi / len(self.categories)
		offset = 2.0 * math.pi / 3.0
		for count in range(len(self.categories)):
			self.colours.append(((2.0 + (math.cos((0.0 * offset) + (step * count)))) / 3.0, (2.0 + (math.cos((1.0 * offset) + (step * count)))) / 3.0, (2.0 + (math.cos((2.0 * offset) + (step * count)))) / 3.0))

		for count in range(len(self.categories) - 1):
			if count % 2 == 0:
				self.colours[count], self.colours[count + 1] = self.colours[count + 1], self.colours[count]
		self.colours.reverse()

		#self.colours = ['#94070a', '#00381f', '#00864b', '#009353', '#00b274', '#65c295', '#8ccfb7', '#bee3d3']

	def annual_stats(self):
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

		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6, 12), dpi=180)

		# Average quantities
		daily = [ totals[category]['quantity'] / self.duration for category in self.categories ]
		print(self.categories)
		print(len(daily))

		bottom = [0]
		for i in range(1, len(daily)):
			bottom.append(bottom[i - 1] + daily[i - 1])

		ax[0].bar([0], daily, bottom=bottom, color=self.colours, width=1.0)

		ax[0].set_ylim(ax[0].get_ylim())

		ax[0].set_xlabel("Daily number\nitems/day")
		ax[0].set_xticklabels([])

		daily_total = total['quantity'] / self.duration
		ax[0].text(0, daily_total, '{:.1f} items'.format(daily_total), ha='center', va='bottom')

		# Average weights
		daily = [ totals[category]['weight'] / self.duration for category in self.categories ]
		print(self.categories)
		print(len(daily))

		bottom = [0]
		for i in range(1, len(daily)):
			bottom.append(bottom[i - 1] + daily[i - 1])

		ax[1].bar([0], daily, bottom=bottom, color=self.colours, width=1.0)

		ax[1].set_ylim(ax[1].get_ylim())

		ax[1].set_xlabel("Daily weight\ng/day")
		ax[1].set_xticklabels([])

		daily_total = total['weight'] / self.duration
		ax[1].text(0, daily_total, '{:.0f} g'.format(daily_total), ha='center', va='bottom')

		# Average prices
		daily = [ totals[category]['price'] / self.duration for category in self.categories ]
		print(self.categories)
		print(len(daily))

		bottom = [0]
		for i in range(1, len(daily)):
			bottom.append(bottom[i - 1] + daily[i - 1])

		ax[2].bar([0], daily, bottom=bottom, color=self.colours, width=1.0)

		ax[2].set_ylim(ax[2].get_ylim())

		ax[2].set_xlabel("Daily price\n€/day")
		ax[2].set_xticklabels([])

		daily_total = total['price'] / self.duration
		ax[2].text(0, daily_total, '{:.2f} €'.format(daily_total), ha='center', va='bottom')

		patches = []
		for count in range(len(self.categories)):
			patches.append(mpatches.Patch(color=self.colours[count], label=self.categories[count].capitalize()))
		patches.reverse()
		fig.legend(handles=patches, loc='right', ncol=1, bbox_to_anchor=(1.3, 0.5))

		fig.suptitle("Daily consumption based on purchases")
		fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
		plt.tight_layout(pad=2.0, w_pad=0.5)
		plt.savefig('temp.png', bbox_inches='tight', transparent=True)
		plt.close()

	def draw_year_graph(self):
		start_date = self.purchases[0]['date']
		end_date = self.purchases[-1]['date']
		value = 'price'
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



		
		filenames = ['temp01.png', 'temp01small.png']
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
			graph.create_stackedareacurve(width, dpi=dpi, filename=filepath)



	def process_inputs(self):
		print("# Calculated inputs")
		print()

		if self.config.latest:
			self.config.year = self.dates[-1].year

		self.file_suffix = ''
		if self.config.suffix:
			self.file_suffix = '-{}'.format(self.config.suffix)
		else:
			if self.config.year and not self.config.latest:
				self.file_suffix='-{}'.format(self.config.year)
			else:
				if not self.config.start_date and not self.config.end_date and not self.config.year and not self.config.latest:
					self.file_suffix='-all'

		if len(self.file_suffix) > 0:
			print('File suffix: "{}"'.format(self.file_suffix))
		else:
			print('File suffix: None')

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
			start_overhang_data = Graphs.get_data_point(self.types, start_pos)
			start_overhang_data = Graphs.scale_data_point(start_overhang_data, factor)
			Graphs.replace_data_point(self.types, start_overhang_data, start_pos)
			# Insert an empty data point at the start
			if factor > 0.0:
				self.dates.insert(start_pos, self.config.start_date)
				empty_data = [0] * len(self.types)
				Graphs.insert_data_point(self.types, empty_data, start_pos)
				end_pos += 1

		# Check whether we need to insert an extra date at the end to accommodate overhang
		if end_pos < len(self.dates):
			factor = 1.0 - ((self.dates[end_pos] - self.config.end_date).days / (self.dates[end_pos] - self.dates[end_pos - 1]).days)
			end_overhang_data = Graphs.get_data_point(self.types, end_pos)
			end_overhang_data = Graphs.scale_data_point(end_overhang_data, factor)
			self.dates.insert(end_pos, self.config.end_date)
			# Insert an extra scaled data point at the end
			Graphs.insert_data_point(self.types, end_overhang_data, end_pos)
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

		print()

	def overview(self):
		if (len(self.dates) > 2):
			print("# Overview")
			print()
			start_date = self.dates[0]
			penultimate_date = self.dates[len(self.dates) - 2]
			end_date = self.dates[len(self.dates) - 1]
			duration = (end_date - start_date).days
			total = sum([sum(wastetype) for wastetype in self.types])
			latest_total = sum([wastetype[len(wastetype) - 1] for wastetype in self.types])
			latest_duration = (end_date - penultimate_date).days

			start_year = start_date.year
			this_year = end_date.year

			year_average = {}
			for year in range(start_year, this_year + 1):
				year_total = 0
				start = 0
				end = 0
				for i in range(len(self.dates)):
					if self.dates[i].year == year:
						if start == 0:
							start = self.dates[i]
							year_pos_start = i
						year_total += sum([wastetype[i] for wastetype in self.types])
						end = self.dates[i]
						year_pos_end = i
				# Add the fractional parts at the start
				if year_pos_start > 0:
					proportion = (datetime.date(year, 1, 1) - self.dates[year_pos_start - 1]).days / (self.dates[year_pos_start] - self.dates[year_pos_start - 1]).days
					year_total -= proportion * sum([wastetype[year_pos_start] for wastetype in self.types])
					start = datetime.date(year, 1, 1)

				# Add the fractional parts at end
				if year_pos_end < len(self.dates) - 1:
					proportion = (datetime.date(year, 12, 31) - self.dates[year_pos_end]).days / (self.dates[year_pos_end + 1] - self.dates[year_pos_end]).days
					year_total += proportion * sum([wastetype[year_pos_end + 1] for wastetype in self.types])
					end = datetime.date(year, 12, 31)

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

	def format_path(self, filename):
		return '{}/{}'.format(self.config.output_dir, filename) if self.config.output_dir else filename

	def plot_graphs(self):
		print("# Plotting data")
		print()

		self.upload = []

		# Stacked line graph for debugging purposes, so don't upload
		filenames = ['waste01{}.png'.format(self.file_suffix), 'waste01small{}.png'.format(self.file_suffix)]
		self.upload = self.upload + filenames
		dpis = [180, 90]
		for filename, dpi in zip(filenames, dpis):
			print("Generating graph '{}' at {} dpi".format(filename, dpi))
			graph = LineGraph()
			graph.dates = self.dates
			graph.types = self.types
			graph.labels = self.labels
			graph.colours = self.colours
			graph.start_date = self.config.start_date
			graph.end_date = self.config.end_date
			filepath = self.format_path(filename)
			graph.create_plot(self.width, dpi=dpi, filename=filepath)

		filenames = ['waste08{}.png'.format(self.file_suffix), 'waste08small{}.png'.format(self.file_suffix)]
		self.upload = self.upload + filenames
		for filename, dpi in zip(filenames, dpis):
			print("Generating graph '{}' at {} dpi".format(filename, dpi))
			graph = Histocurve()
			graph.dates = self.dates
			graph.types = self.types
			graph.labels = self.labels
			graph.colours = self.colours
			graph.start_date = self.config.start_date
			graph.end_date = self.config.end_date
			filepath = self.format_path(filename)
			graph.create_stackedareacurve(self.width, dpi=dpi, filename=filepath)

		for i in range(0, len(graph.types)):
			filenames = []
			filenames.append("waste-detail0{}-{}{}.png".format(i, self.labels[i].lower(), self.file_suffix))
			filenames.append("waste-detail0{}small-{}{}.png".format(i, self.labels[i].lower(), self.file_suffix))
			self.upload = self.upload + filenames
			for filename, dpi in zip(filenames, dpis):
				print("Generating graph '{}' at {} dpi".format(filename, dpi))
				graph = Histogram()
				graph.dates = self.dates
				graph.start_date = self.config.start_date
				graph.end_date = self.config.end_date
				filepath = self.format_path(filename)
				graph.create_histogram(self.width, dpi=dpi, filename=filepath, data=self.types[i], ylimit=0, colour=self.colours[i], title=self.labels[i])

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
		self.plot_graphs()
		self.ftp_upload()

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

	def create_stackedareacurve(self, width, dpi, filename):
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

		#ax[0].set_ylabel("Quantity purchased (g / day)")
		ax[0].set_ylabel("Quantity purchased (€ / day)")
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

		#fig.suptitle("Weight of household purchases")
		fig.suptitle("Cost of household purchases")
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
#consumption.parse_arguments()
#consumption.execute_config()
consumption.load_category_data('categories.csv', 'purchases.csv')
consumption.annual_stats()
consumption.draw_year_graph()



print("All done")


