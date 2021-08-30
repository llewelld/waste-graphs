# Waste graph generating script

This python script is used for generating graphs from waste weight data. The script accepts an optional configuration file in JSON format and a required data file in CSV format. It ouputs PNG format graphs.

The script is used to generate the graphs for https://www.flypig.co.uk/waste

For the details of how the histocurve is generated see https://www.flypig.co.uk/?page=list&list_id=641&list=blog

## Command line arguments

The script can be controlled either using command line parameters, or using the values in the configuration JSON file, or both, as explained below.

```
usage: graphs.py [-h] [--config CONFIG] [--start START] [--end END] [--input INPUT] [--year YEAR] [--latest] [--all] [--suffix SUFFIX] [--ftp FTP]
                 [--username USERNAME] [--password PASSWORD]

Generate graphs about waste output

optional arguments:
  -h, --help           show this help message and exit
  --config CONFIG      read config values from file
  --start START        start date to plot from
  --end END            end date to plot to
  --input INPUT        CSV data file to load
  --outdir OUTDIR      Directory to store the generated PNG graphs
  --year YEAR          plot graphs for a given year; overrides start and end arguments
  --latest             plot the most recent year; overrides start, end and year arguments
  --all                plot all values; overrides other time values
  --suffix SUFFIX      filename suffix to add; this will be chosen based on the input values if not explicitly provided
  --ftp FTP            location to use for FTP upload in the form: server/path
  --username USERNAME  username to use for FTP upload
  --password PASSWORD  password to use for FTP upload
```

## Data CSV file format

The data should be stored in a CSV file, with the default name of `recycling.csv`, although an alternative filename can be provided using the `--input` argument.

The CSV file should be in the following format.

```
Header row
Data raw
Data row
```

Where `Header row` contains the ten column headers separated by commas like this:
```
Date,Paper,Card,Glass,Metal,Returnables,Compost,Plastic,General,Notes
```

Each data row contains ten entries. The first is a date in the format `DD/MM/YY`; the last is a string encapsulated by quote marks, while the intermediate values are all integers (weights measured in grams) corresponding to the waste categorisations from the header line.
```
<DD/MM/YY>, <int>, <int>, <int>, <int>, <int>, <int>, <int>, <int>, <int>, "<string>"
```

Here's the first three lines of an example file.
```
Date,Paper,Card,Glass,Metal,Returnables,Compost,Plastic,General,Notes
11/08/19,0,0,0,0,0,0,0,0,"Not a reading"
18/08/19,221,208,534,28,114,584,0,426,""
```

## Config JSON file format

The configuration file is optional and is only used if the `--config` parameter is passed at the command line. The file is a dictionary where each of the entries corresponds to one of the command line parameters described above.

In addition to these entries, a further entry `repeat` may be included. This should be an array where each entry is in a similar format to the dictionary described in the previous paragraph (i.e. containing entries that match the command line parameters).

If there is no `repeat` entry, the script is executed a single time using the values contained in the script. If a `repeat` array is included, the script will be run multiple times, once for each entry in the array.

On each execution of the script, configuration options are layered on top of each other based on their precedence. First the options in the root of the config file are applied, followed by the entries in the repeat entry, followed by the command line parameters. Thus the command line parameters have the highest precedence and the root entries in the config file have the least predendence.

The following shows an example configuration file `config-example.json`:
```
{
	"input": "recycling-example.csv",
	"repeat": [
		{
			"all": 1
		},
		{
			"latest": 1
		}
	]
}
```

The example config can be executed like this.
```
./graphs.py --config config-example.json
```

In this case the script will cycle twice. On both cycles the `recycling-example.csv` script will be used for the input data. On the first cycle the graphs for the full set of data spanning multiple years will be output. On the second cycle the graphs just for the most recent year will be output.

## Contact

Name: David Llewellyn-Jones

Email: david@flypig.co.uk

Website: https://www.flypig.co.uk

Visit https://www.flypig.co.uk/waste for more info.

