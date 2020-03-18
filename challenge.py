import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import db_password
import sys
import time
import psycopg2

file_dir = '/Users/sagpalo/Documents/1_BootCamp_Class/Movies-ETL'


#This function will take in new arguments - wiki,kaggle,ratings
def movie_files(wiki_movies_raw,kaggle_metadata,ratings):
	#Import files
	with open(f'{file_dir}/{wiki_movies_raw}', mode='r') as file:
		wiki_movies_raw = json.load(file)
	kaggle_metadata = pd.read_csv(f'{file_dir}/{kaggle_metadata}',low_memory=False)
	ratings = pd.read_csv(f'{file_dir}/{ratings}')
	
	#modify our JSON data by restricting it to only those entries that have a director and an IMDb link
	#and are not TV episodes
	wiki_movies = [movie for movie in wiki_movies_raw
			   if ('Director' in movie or 'Directed by' in movie)
				   and 'imdb_link' in movie
				   and 'No. of episodes' not in movie]
	#Create wiki dataframe
	wiki_movies_df = pd.DataFrame(wiki_movies)

	#create our function to clean our movie data
	def clean_movie(movie):
		movie = dict(movie) #create a non-destructive copy
		alt_titles = {}
	# combine alternate titles into one list
		for key in ['Also known as','Arabic','Cantonese','Chinese','French',
					'Hangul','Hebrew','Hepburn','Japanese','Literally',
					'Mandarin','McCune-Reischauer','Original title','Polish',
					'Revised Romanization','Romanized','Russian',
					'Simplified','Traditional','Yiddish']:
			if key in movie:
				alt_titles[key] = movie[key]
				movie.pop(key)
		if len(alt_titles) > 0:
			movie['alt_titles'] = alt_titles

	# merge column names
		def change_column_name(old_name, new_name):
			if old_name in movie:
					movie[new_name] = movie.pop(old_name)
		change_column_name('Adaptation by', 'Writer(s)')
		change_column_name('Country of origin', 'Country')
		change_column_name('Directed by', 'Director')
		change_column_name('Distributed by', 'Distributor')
		change_column_name('Edited by', 'Editor(s)')
		change_column_name('Length', 'Running time')
		change_column_name('Original release', 'Release date')
		change_column_name('Music by', 'Composer(s)')
		change_column_name('Produced by', 'Producer(s)')
		change_column_name('Producer', 'Producer(s)')
		change_column_name('Productioncompanies ', 'Production company(s)')
		change_column_name('Productioncompany ', 'Production company(s)')
		change_column_name('Released', 'Release Date')
		change_column_name('Release Date', 'Release date')
		change_column_name('Screen story by', 'Writer(s)')
		change_column_name('Screenplay by', 'Writer(s)')
		change_column_name('Story by', 'Writer(s)')
		change_column_name('Theme music composer', 'Composer(s)')
		change_column_name('Written by', 'Writer(s)')

		return movie

	#make a list of cleaned movies with a list comprehension
	clean_movies = [clean_movie(movie) for movie in wiki_movies]

	#create dataframe
	wiki_movies_df = pd.DataFrame(clean_movies)

	#extract the IMDb ID from the IMDb link
	wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')

	#drop duplicates
	wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

	#remove null values
	wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
	wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

	#drop missing values in box office column
	box_office = wiki_movies_df['Box office'].dropna()

	#ensure box office data is entered as string type
	box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

	#create forms to parse box data
	form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
	form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

	#for values given as a range
	box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

	#create new function to turn the extracted values into a numeric value
	def parse_dollars(s):
	# if s is not a string, return NaN
		if type(s) != str:
			return np.nan

		# if input is of the form $###.# million
		if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

		# remove dollar sign and " million"
			s = re.sub('\$|\s|[a-zA-Z]','', s)

		# convert to float and multiply by a million
			value = float(s) * 10**6

		# return value
			return value

		# if input is of the form $###.# billion
		elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

		# remove dollar sign and " billion"
			s = re.sub('\$|\s|[a-zA-Z]','', s)

		# convert to float and multiply by a billion
			value = float(s) * 10**9

		# return value
			return value

		# if input is of the form $###,###,###
		elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

		# remove dollar sign and commas
			s = re.sub('\$|,','', s)

		# convert to float
			value = float(s)

		# return value
			return value

		else:
			return np.nan

	#extract values from box office and apply parse_dollars function
	wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

	#drop box office column
	wiki_movies_df.drop('Box office', axis=1, inplace=True)

	#parse budget data
	#create budget variable
	budget = wiki_movies_df['Budget'].dropna()

	#convert any lists to string
	budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

	#remove any values between a dollar sign and a hyphen
	budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

	#extract values from box budget and apply parse_dollars function
	wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

	#drop budget column
	wiki_movies_df.drop('Budget', axis=1, inplace=True)

	#parse release date
	#make a variable that holds the non-null values of Release date in the DataFrame, converting lists to strings
	release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

	#parse different forms
	date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
	date_form_two = r'\d{4}.[01]\d.[123]\d'
	date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
	date_form_four = r'\d{4}'

	#parse release_date using to_datetime() method
	wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

	#parse running time
	#make a avriable that holds a non-null values of running time in the DataFrame, converting lists to strings
	running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

	#extract running time values
	running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

	#convert to numeric values
	running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

	#apply a function that will convert the hour capture groups and minute capture groups to minutes
	# if the pure minutes capture group is zero, and save the output to wiki_movies_df
	wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

	#drop Running time from the dataset
	wiki_movies_df.drop('Running time', axis=1, inplace=True)


	#clean Kaggle metadata
	#keep rows where the adult column is False, and then drop the adult column
	kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

	#create boolean column and assign back to video
	kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

	#use the to_numeric() method from Pandas for numeric columns
	kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
	kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
	kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

	#convert release_date using to_datetime()
	kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

	#Ratings Data
	#convert timestamp using to_datetime() method and assign to timestamp column
	ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

	#merge wikipedia and kaggle metadata
	movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

	#drop outliers (Holiday and From here to Eternity)
	movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

	#drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
	movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

	#create function that fills in missing data for a column pair and then drops the redundant column
	def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
		df[kaggle_column] = df.apply(
			lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
			, axis=1)
		df.drop(columns=wiki_column, inplace=True)

	#run the function for the three column pairs that we decided to fill in zeros
	fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
	fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
	fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

	#check that there aren’t any columns with only one value
	try:
		for col in movies_df.columns:
			lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
			value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
			num_values = len(value_counts)
			if num_values == 1:
				print(col)
	except:
		print('No columns to drop')

	#reorder the columns
	movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
					   'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
					   'genres','original_language','overview','spoken_languages','Country',
					   'production_companies','production_countries','Distributor',
					   'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
					  ]]

	#rename the columns to be consistent
	movies_df.rename({'id':'kaggle_id',
				  'title_kaggle':'title',
				  'url':'wikipedia_url',
				  'budget_kaggle':'budget',
				  'release_date_kaggle':'release_date',
				  'Country':'country',
				  'Distributor':'distributor',
				  'Producer(s)':'producers',
				  'Director':'director',
				  'Starring':'starring',
				  'Cinematography':'cinematography',
				  'Editor(s)':'editors',
				  'Writer(s)':'writers',
				  'Composer(s)':'composers',
				  'Based on':'based_on'
				 }, axis='columns', inplace=True)

	#transform and merge rating data
	#use a groupby on the “movieId” and “rating” columns and take the count for each group, rename the “userId” column to “count.”
	#pivot this data so that movieId is the index, the columns will be all the rating values, and the rows will be the counts for each rating value
	rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count().rename({'userId':'count'}, axis=1).pivot(index='movieId',columns='rating', values='count')

	#rename the columns so they’re easier to understand
	rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

	#use a left merge, since we want to keep everything in movies_df
	movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

	#fill in missing values
	movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

	#connect Pandas and SQL
	#install psycopg2-binary if not installed already

	#create database engine
	db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
	engine = create_engine(db_string)

	# You’ll need to remove the existing data from SQL, but keep the empty tables.
	#movies
	try:
		connection = psycopg2.connect(db_string)
		cursor = connection.cursor()
		delete_movie_data = "DELETE FROM movies"
		cursor.execute(delete_movie_data)
		connection.commit()
		print("Records deleted successfully from movies table")

	except:
		print("There's an error in the delete operation")
		
	
	#ratings
	try:
		connection = psycopg2.connect(db_string)
		cursor = connection.cursor()
		delete_ratings_data = "DELETE FROM ratings"
		cursor.execute(delete_ratings_data)
		connection.commit()
		print("Records deleted successfully from ratings table")

	except:
		print("There's an error in the delete operation")


	#import movie data
	movies_df.to_sql(name='movies', con=engine, if_exists='append')

	#import ratings data
	rows_imported = 0
	# get the start_time from time.time()
	start_time = time.time()
	for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):
		print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
		data.to_sql(name='ratings', con=engine, if_exists='append')
		rows_imported += len(data)

		# add elapsed time to final print out
		print(f'Done. {time.time() - start_time} total seconds elapsed')


#execute function
movie_files('wikipedia.movies.json', 'movies_metadata.csv', 'ratings.csv')








