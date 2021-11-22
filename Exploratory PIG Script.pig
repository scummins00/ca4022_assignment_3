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

-- Checking how many unique companies are in the dataset (there are 1,635 distinct values)
companies = FOREACH labelled_data GENERATE company;
distinct_companies = DISTINCT companies;
grouped_companies = GROUP distinct_companies ALL;
distinct_company_count = FOREACH grouped_companies GENERATE COUNT(distinct_companies);
DUMP distinct_company_count;

-- Converting company names to uppercase to avoid duplicates from camel-case and lowercase entries
uppercase_companies_data = 
	FOREACH labelled_data
	GENERATE
		timestamp, 
		UPPER(company) as company, 
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

-- Checking how many unique companies remain in the dataset 
-- (There are now 1,104 distinct values, meaning 531 entries were duplicates)
companies = FOREACH uppercase_companies_data GENERATE company;
distinct_companies = DISTINCT companies;
grouped_companies = GROUP distinct_companies ALL;
distinct_company_count = FOREACH grouped_companies GENERATE COUNT(distinct_companies);
DUMP distinct_company_count;

STORE uppercase_companies_data INTO 'cleaned_salary_data' using PigStorage('|');