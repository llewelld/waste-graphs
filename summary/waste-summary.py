#!/usr/bin/python3

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Summary:
	rows = []
	data = {}
	skiprows = ['year', 'duration', 'total']
	colours = ['#94070a', '#00381f', '#00864b', '#009353', '#00b274', '#65c295', '#8ccfb7', '#bee3d3']

	def load_data(self, input_file):
		with open(input_file, 'rt') as data:
			reader = csv.reader(data, delimiter=',', quotechar='"')

			for row in reader:
				key = row[0].lower()
				if key == 'year':
					data = row[1:]
				else:
					data = [float(val) for val in row[1:]]
				self.data[key] = data
				if not key in self.skiprows:
					self.rows.append(key)
				print('{}: {}'.format(key, data))

	@staticmethod
	def table_header(key, data):
		print('\t\t\t<th>{}</th>'.format(key.title()))
		for cell in data:
			print('\t\t\t<th>{}</th>'.format(cell))

	@staticmethod
	def table_row(key, data):
		print('\t\t\t<td>{}</td>'.format(key.title()))
		for cell in data:
			print('\t\t\t<td align="right">{:0.2f}</td>'.format(cell))

	def output_table_average(self):
		print('<table align="center" border="1" cellpadding="4" cellspacing="0">')
		print('\t<tbody>')
		print('\t\t<tr align="left">')
		self.table_header('year', self.data['year'])
		print('\t\t</tr>')
		for key in self.rows:
			print('\t\t<tr>')
			self.table_row(key, self.data[key])
			print('\t\t</tr>')
		print('\t\t<tr>')
		self.table_row('total', self.data['total'])
		print('\t\t</tr>')
		print('\t</tbody>')
		print('</table>')

	def output_table_total(self):
		print('<table align="center" border="1" cellpadding="4" cellspacing="0">')
		print('\t<tbody>')
		print('\t\t<tr align="left">')
		self.table_header('year', self.data['year'])
		print('\t\t</tr>')
		for key in self.rows:
			row = [cell * 365.25 / 1000.0 for cell in self.data[key]]
			print('\t\t<tr>')
			self.table_row(key, row)
			print('\t\t</tr>')
		print('\t\t<tr>')
		row = [cell * 365.25 / 1000.0 for cell in self.data['total']]
		self.table_row('total', row)
		print('\t\t</tr>')
		print('\t</tbody>')
		print('</table>')

	def plot_sub_stacked(self, ax, quantities):
		bottom = [0]
		for i in range(1, len(quantities)):
			bottom.append(bottom[i - 1] + quantities[i - 1])

		ax.bar([0], quantities, bottom=bottom, color=self.colours, width=1.0)
		#ax.set_ylim(ax.get_ylim())
		ax.set_xticklabels([])

	def output_graph_average(self):
		filenames = ['summary01-2022.png', 'summary01small-2022.png']
		dpis = [180, 90]
		for filename, dpi in zip(filenames, dpis):
			print("Generating graph '{}' at {} dpi".format(filename, dpi))
			fig, ax = plt.subplots(nrows=1, ncols=len(self.data['year']), figsize=(6, 12), dpi=dpi)
			
			for pos in range(len(self.data['year'])):
				year = self.data['year'][pos]
				
				quantities = [self.data[key][pos] for key in self.rows]
				self.plot_sub_stacked(ax[pos], quantities)
				ax[pos].set_xlabel(year)
				quantities_total = self.data['total'][pos]
				ax[pos].text(0, quantities_total, '{:.2f} g'.format(quantities_total), ha='center', va='bottom')
				ax[pos].set_ylim(bottom=0, top=350)

			patches = []
			for count in range(len(self.rows)):
				patches.append(mpatches.Patch(color=self.colours[count], label=self.rows[count].capitalize()))
			patches.reverse()
			fig.legend(handles=patches, loc='right', ncol=1, bbox_to_anchor=(1.3, 0.5))

			fig.suptitle("Average daily waste output")
			fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
			plt.tight_layout(pad=2.0, w_pad=0.5)
			plt.savefig(filename, bbox_inches='tight', transparent=True)
			plt.close()

	def output_graph_proportion(self):
		filenames = ['summary02-2022.png', 'summary02small-2022.png']
		dpis = [180, 90]
		for filename, dpi in zip(filenames, dpis):
			print("Generating graph '{}' at {} dpi".format(filename, dpi))
			fig, ax = plt.subplots(nrows=1, ncols=len(self.data['year']), figsize=(6, 12), dpi=dpi)
			
			for pos in range(len(self.data['year'])):
				year = self.data['year'][pos]
				
				total = self.data['total'][pos] if self.data['total'][pos] > 1 else 1
				quantities = [100.0 * self.data[key][pos] / total for key in self.rows]
				self.plot_sub_stacked(ax[pos], quantities)
				ax[pos].set_xlabel(year)
				quantities_total = sum(quantities)
				ax[pos].text(0, quantities_total, '{:.0f}%'.format(quantities_total), ha='center', va='bottom')

			patches = []
			for count in range(len(self.rows)):
				patches.append(mpatches.Patch(color=self.colours[count], label=self.rows[count].capitalize()))
			patches.reverse()
			fig.legend(handles=patches, loc='right', ncol=1, bbox_to_anchor=(1.3, 0.5))

			fig.suptitle("Average daily proportional waste output")
			fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
			plt.tight_layout(pad=2.0, w_pad=0.5)
			plt.savefig(filename, bbox_inches='tight', transparent=True)
			plt.close()

summary = Summary()
summary.load_data('waste-summary.csv')
summary.output_table_average()
print()
summary.output_table_total()
summary.output_graph_average()
summary.output_graph_proportion()

