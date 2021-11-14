-- Setting up CSVLoader to allow for commas in movie titles
DEFINE CSVLoader org.apache.pig.piggybank.storage.CSVLoader();

-- Importing the datasets
headed_data = LOAD 'Levels_Fyi_Salary_Data.csv' 
	USING CSVLoader() 
	AS (
		timestamp:chararray, 
		company:chararray, 
		level:chararray,
		title:chararray, 
		totalyearlycompensation:int, 
		location:chararray,
		yearsofexperience:float, 
		yearsatcompany:float, 
		tag:chararray,
		basesalary:int, 
		stockgrantvalue:int, 
		bonus:int,
		gender:chararray, 
		otherdetails:chararray, 
		cityid:chararray,
		dmaid:chararray, 
		rownumber:chararray, 
		masters_degree:chararray,
		bachelors_degree:chararray, 
		doctorate_degree:chararray, 
		highschool:chararray,
		some_college:chararray,
		race_asian:chararray,
		race_white:chararray, 
		race_two_or_more:chararray, 
		race_black:chararray,
		race_hispanic:chararray,
		race:chararray, 
		education:chararray);

-- Removing the header row
data = FILTER headed_data BY timestamp != 'timestamp';

-- Removing the one-hot-encoded data
labelled_data = 
	FOREACH data
	GENERATE
		timestamp, 
		company, 
		level,
		title, 
		totalyearlycompensation, 
		location,
		yearsofexperience, 
		yearsatcompany, 
		tag,
		basesalary, 
		stockgrantvalue, 
		bonus,
		gender, 
		otherdetails, 
		cityid,
		dmaid, 
		race, 
		education;

ltd = limit labelled_data 10;
dump ltd;