-- Setting up CSVLoader to allow for commas in movie titles
DEFINE CSVLoader org.apache.pig.piggybank.storage.CSVLoader();

-- Importing the datasets
headed_data = LOAD 'data/Levels_Fyi_Salary_Data.csv' 
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

-- Checking how many unique companies are in the dataset (there are 1,633 distinct values)
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
-- (There are now 1,102 distinct values, meaning 531 entries were duplicates)
companies = FOREACH uppercase_companies_data GENERATE company;
distinct_companies = DISTINCT companies;
grouped_companies = GROUP distinct_companies ALL;
distinct_company_count = FOREACH grouped_companies GENERATE COUNT(distinct_companies);
DUMP distinct_company_count;

--Find the average salary
has_salary = FILTER uppercase_companies_data BY basesalary != 0;
avg_salary_grouped = GROUP has_salary ALL;
avg_salary = FOREACH avg_salary_grouped GENERATE ROUND(AVG(has_salary.basesalary)) as avg;

--Give anyone with a salary of 0 the average salary
imputed_data =
	FOREACH uppercase_companies_data
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
		(basesalary > 0 ? basesalary : avg_salary.avg) AS imputed_salary, 
		stockgrantvalue, 
		bonus,
		gender, 
		otherdetails, 
		cityid,
		dmaid, 
		race, 
		education;

-- Removing instances of pipe bar 
replaced_data = 
	FOREACH imputed_data 
	GENERATE 
		REPLACE(timestamp,'|',''), 
		REPLACE(company,'|',''), 
		REPLACE(level,'|',''), 
		REPLACE(title,'|',''), 
		totalyearlycompensation,
		REPLACE(location,'|',''), 
		yearsofexperience,
		yearsatcompany,
		REPLACE(tag,'|',''), 
		imputed_salary,
		stockgrantvalue,
		bonus, 
		REPLACE(gender,'|',''), 
		REPLACE(otherdetails,'|',''), 
		REPLACE(cityid,'|',''), 
		REPLACE(dmaid,'|',''), 
		REPLACE(race,'|',''), 
		REPLACE(education,'|','');

STORE replaced_data INTO 'data/replaced_salary_data' USING PigStorage('|', 'schema');